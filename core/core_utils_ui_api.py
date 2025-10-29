import os
import re
import stat
import time
import logging
import urllib.parse
from datetime import datetime
from pathlib import Path
from difflib import SequenceMatcher
from urllib.parse import urlparse

import requests
import pandas as pd
from bs4 import BeautifulSoup
from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# =========================
# 경로/환경 유틸
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RES_DIR = PROJECT_ROOT / "resources"
LOG_DIR = PROJECT_ROOT / "data" / "log"
LOG_DIR.mkdir(parents=True, exist_ok=True)

def respath(*names: str) -> Path:
    """resources/ 내부 경로 조합"""
    return RES_DIR.joinpath(*names)

def find_resource(*candidates: str) -> Path:
    """
    resources/ 아래에서 파일명을 재귀 탐색하여 첫 매치 경로를 반환.
    - candidates: 우선순위대로 탐색할 파일명(확장자 포함)
    - 찾지 못하면 RES_DIR/첫후보 경로를 반환(존재X)
    """
    for name in candidates:
        for p in RES_DIR.rglob(name):
            if p.is_file():
                return p
    return RES_DIR / candidates[0]

def _safe_read_excel(path: Path, required_col: str | None = None,
                     default_list=None, create_template=False):
    """
    엑셀 안전 로딩:
      - 파일이 없거나, 컬럼이 없거나, 로딩 실패 시 기본값 반환
      - create_template=True면 헤더만 있는 템플릿 생성 시도
    """
    default_list = [] if default_list is None else default_list
    try:
        if not path.exists():
            logging.warning(f"[WARN] 리소스 파일 없음: {path}")
            if create_template and required_col:
                try:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame(columns=[required_col]).to_excel(path, index=False)
                    logging.info(f"[INFO] 템플릿 생성: {path} (컬럼: {required_col})")
                except Exception as te:
                    logging.warning(f"[WARN] 템플릿 생성 실패: {path} ({te})")
            return default_list
        df = pd.read_excel(path)
        if required_col and required_col not in df.columns:
            logging.warning(f'[WARN] 리소스 컬럼 없음: {path} (필요: "{required_col}", 실제: {list(df.columns)})')
            return default_list
        if required_col:
            return (df[required_col].dropna().astype(str).str.strip().tolist())
        return df
    except Exception as e:
        logging.warning(f"[WARN] 리소스 로딩 실패: {path} ({e})")
        return default_list

# =========================
# 환경 변수(.env) 로드
# =========================
load_dotenv(PROJECT_ROOT / ".env")
NAVER_CLIENT_ID = os.environ.get("NAVER_CLIENT_ID", "")
NAVER_CLIENT_SECRET = os.environ.get("NAVER_CLIENT_SECRET", "")

# =========================
# 로깅 설정
# =========================
today = datetime.now().strftime("%y%m%d")
log_path = LOG_DIR / f"로그_{today}.txt"

logger = logging.getLogger()
logger.setLevel(logging.INFO)
# 중복 핸들러 방지
if not logger.handlers:
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

def log(msg: str, index: int | None = None):
    prefix = f"[{index+1:03d}] " if index is not None else ""
    logger.info(f"{prefix}{msg}")

# 형태소 분석기
okt = Okt()

# =========================
# 리소스 로딩
# =========================

# (1) 수집 제외 도메인
excluded_domains_file = find_resource("수집 제외 도메인 주소.xlsx",
                                      "(언진) 수집 제외 도메인 주소_공식 블로그.xlsx")
excluded_domains = _safe_read_excel(
    excluded_domains_file, required_col="제외 도메인 주소",
    default_list=[], create_template=False
)

# (2) 신탁언론 OID 목록
news_oid_file  = find_resource("네이버뉴스 신탁언론 oid.xlsx")
sport_oid_file = find_resource("네이버스포츠 신탁언론 oid.xlsx")
ent_oid_file   = find_resource("네이버엔터 신탁언론 oid.xlsx")


def _load_oid_set(path: Path) -> set[str]:
    """
    엑셀에서 OID 컬럼을 읽어 3자리 문자열 세트로 반환.
    컬럼명이 대소문자/공백 혼재 시에도 대응.
    """
    if not path.exists():
        log(f"⚠️ OID 파일 없음: {path}")
        return set()
    try:
        df = pd.read_excel(path)
        # 후보 컬럼명들
        cand = [c for c in df.columns if str(c).strip().lower() in ("oid", "o id", "oid코드")]
        if not cand:
            log(f"⚠️ OID 컬럼 없음: {path} / 실제컬럼: {list(df.columns)}")
            return set()
        col = cand[0]
        return set(
            df[col].dropna().astype(int).astype(str).map(lambda s: s.zfill(3))
        )
    except Exception as e:
        log(f"⚠️ OID 로딩 실패: {path} ({e})")
        return set()

trusted_news_oids       = _load_oid_set(news_oid_file)
trusted_sports_oids     = _load_oid_set(sport_oid_file)
trusted_entertain_oids  = _load_oid_set(ent_oid_file)

log(f"📦 신탁 OID(뉴스/스포츠/연예) 크기: {len(trusted_news_oids)}/{len(trusted_sports_oids)}/{len(trusted_entertain_oids)}")

# 도메인 화이트리스트 불러옴
trusted_domains_file = find_resource("매체사_도메인_정보.xlsx")
trusted_domains = _safe_read_excel(
    trusted_domains_file, required_col="도메인", default_list=[]
)
log(f"📦 도메인 화이트리스트 크기: {len(trusted_domains)}")

# (임시 완화용 플래그)
_USE_STRICT_FILTER = True
if (len(trusted_news_oids) + len(trusted_sports_oids) + len(trusted_entertain_oids) == 0) and len(trusted_domains) == 0:
    log("⚠️ 모든 화이트리스트가 비어 있음 → 임시로 필터 완화", None)
    _USE_STRICT_FILTER = False

# =========================
# 텍스트 전처리/쿼리 생성
# =========================

def clean_text(text, preserve_newline=False):
    """일반적인 노이즈(이모지/특수문자/연속공백 등) 제거.
       preserve_newline=True면 줄바꿈은 유지."""
    if not isinstance(text, str):
        text = str(text)
    if text.strip().lower() == "nan":
        return ""
    patterns = [
        r"Video Player", r"Video 태그를 지원하지 않는 브라우저입니다\.",
        r"\d{2}:\d{2}", r"[01]\.\d{2}x", r"출처:\s?[^\n]+", r"/\s?\d+\.?\d*"
    ]
    for p in patterns:
        text = re.sub(p, "", text)
    text = re.sub(r"[ㅋㅎㅠㅜ]+", "", text)
    text = re.sub(r"[!?~\.,\-#]{2,}", "", text)
    text = re.sub(r"&[a-z]+;|&#\d+;", "", text)
    text = re.sub(r"[\\\xa0\u200b\u3000\u200c_x000D_]", " ", text)
    if preserve_newline:
        # 줄바꿈은 살리고 과도한 공백만 정리
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
    else:
        return re.sub(r"\s+", " ", text).strip()

def extract_keywords(text, num_keywords=5):
    nouns = okt.nouns(text)
    return " ".join(nouns[:num_keywords])

def extract_first_sentences(text):
    """첫 단락의 첫 문장 / 두 번째 단락의 첫 문장 / 마지막 단락의 마지막 문장"""
    paras = re.split(r"\n{2,}", (text or "").strip())
    def _split(p): return re.split(r'(?<=[.!?])(?=\s|[가-힣])', p.strip()) if p else [""]
    first  = _split(paras[0])[0] if len(paras) > 0 else ""
    second = _split(paras[1])[0] if len(paras) > 1 else ""
    last   = _split(paras[-1])[-1].strip() if len(paras) > 0 else ""
    return first, second, last

MAX_QUERY_LENGTH = 100

def _sentences(text: str) -> list[str]:
    """간단한 문장 분리(마침표/물음표/느낌표 뒤 공백 기준)"""
    if not text:
        return []
    parts = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    if len(parts) <= 1:
        parts = [s.strip() for s in re.split(r'(?<=[\.\?\!]|다\)|다]|다»|다”|다’|요\)|요]|요»|요”|요’)', text) if s.strip()]
    return parts

def _sanitize_for_query(s: str) -> str:
    """연속 공백 정리 + 길이 컷(숫자/고유명사 보존)"""
    s = re.sub(r'\s+', ' ', (s or '').strip())
    return s[:MAX_QUERY_LENGTH]

def _strong_phrase(s: str) -> str:
    """정확일치 검색을 위해 따옴표 감싸기(너무 짧으면 그대로)"""
    s = s.strip()
    return f"\"{s}\"" if len(s) >= 8 else s

def generate_search_queries(title, first, second, last, press, *, full_content: str = None):
    """
    제목/문장/언론사 조합 + 본문 '첫 문장부터 2~3문장'을 정확일치 쿼리
    최종 상위 6개 반환
    """
    # 1) 기존 베이스 쿼리 구성
    title_clean  = _sanitize_for_query(clean_text(title))
    first_clean  = _sanitize_for_query(clean_text(first))
    second_clean = _sanitize_for_query(clean_text(second))
    last_clean   = _sanitize_for_query(clean_text(last))
    press_clean  = _sanitize_for_query(clean_text(press))
    keywords     = _sanitize_for_query(extract_keywords(title_clean))

    base_queries = list(filter(None, [
        title_clean,
        (f"{keywords} {press_clean}").strip() if keywords and press_clean else "",
        first_clean, second_clean, last_clean
    ]))

    # 2) 본문 '첫 문장부터 2~3문장' 정확일치 쿼리 추가
    head_sents_queries = []
    if full_content:
        full_clean = clean_text(full_content)
        sents = _sentences(full_clean)[:3]   # 첫 2~3문장(최대 3문장)
        for s in sents:
            q = _sanitize_for_query(s)
            if q:
                head_sents_queries.append(_strong_phrase(q))

    # 3) 합치기 + 중복 제거 + 간단 스코어로 상위 6개 선별
    merged = base_queries + head_sents_queries

    seen, unique = set(), []
    for q in merged:
        if q and q not in seen:
            seen.add(q)
            unique.append(q)

    # 정보량 높은 쿼리를 우선(길이, 숫자 포함, 정확일치 가점)
    def score(q: str) -> tuple:
        has_num = int(bool(re.search(r'\d', q)))
        is_exact = 1 if q.startswith('"') else 0
        return (len(q), has_num, is_exact)

    unique.sort(key=score, reverse=True)
    return unique[:6]

# =========================
# 네이버 링크 OID 추출/필터
# =========================

def extract_oid_from_naver_url(link: str) -> str | None:
    """네이버 뉴스/스포츠/연예 URL에서 OID(3자리) 추출"""
    path = urlparse(link).path
    # 데스크톱
    m = re.search(r"/article/(\d{3})/\d+", path)
    if m: return m.group(1)
    # 모바일
    m = re.search(r"/mnews/article/(\d{3})/\d+", path)
    if m: return m.group(1)
    return None

def is_excluded(url: str) -> bool:
    """수집 제외 도메인 포함 여부"""
    return any(domain for domain in excluded_domains if domain and domain in url)

# =========================
# 본문 추출 셀렉터/폴백 크롤링
# =========================

# 언론사 도메인별 주요 본문 셀렉터 맵
selector_map = {
    "n.news.naver.com": "article#dic_area",
    "m.sports.naver.com": "div._article_content",
    "m.entertain.naver.com": "article#comp_news_article div._article_content",

    "edaily.co.kr": "div.news_body", # 1 이데일리
    "mt.co.kr": "div#textBody", # 2 머니투데이
    "fnnews.com": "div#article_content",  # 3 파이낸셜뉴스
    "khan.co.kr": "div#articleBody", # 4 경향신문
    "sedaily.com": "div.article_view", # 5 서울경제
    "dailian.co.kr": "div.article", # 6 데일리안
    "news.bizwatch.co.kr": "div.news_body.new_editor", # 7 비즈워치
    "asiae.co.kr": "div#txt_area",  # 8 아시아경제
    "kmib.co.kr": "div#articleBody", # 9 국민일보
    "biz.heraldcorp.com": "article#articleText", #10 헤럴드경제
    "newspim.com": "div#news-contents", #11 뉴스핌
    "hani.co.kr": "div.article-text", #12 한겨레
    "nocutnews.co.kr": "div#pnlContent", #13 노컷뉴스
    "ytn.co.kr": "div#CmAdContent",  #14 YTN
    "segye.com": "div#article_txt", #15 세계일보
    #"hankookilbo.com": "div.col-main", #16 한국일보
    "seoul.co.kr": "div.viewContent.body18.color700", #17 서울신문
    "imbc.com": "div.news_txt", #18 MBC
    "cctimes.kr": "div#article-view-content-div", #19 충청타임즈
    "busan.com": "div.article_content", #20 부산일보
    "sbs.co.kr": "div.text_area", #21 SBS
    "kbs.co.kr": "div#cont_newstext", #22 KBS
    "etoday.co.kr": "div.articleView", #23 이투데이
    "breaknews.com": "div#CLtag", #24 BreakNews
    "koreaherald.com": "article#articleText", #25 코리아헤럴드
    "incheonilbo.com": "article#article-view-content-div", #26 인천일보
    "etnews.com": "div#articleBody", #27 전자신문
    "kookje.co.kr": "div.news_article", #28 국제신문
    "ajunews.com": "div#articleBody", #29 아주경제
    "imaeil.com": "div#articlebody", #30 매일신문
    "kyeonggi.com": "div.article_cont_wrap", #31 경기일보
    "ggilbo.com": "article.article-veiw-body", #32 금강일보
    "domin.co.kr": "div#article-view-content-div",#33 전북도민일보
    "asiatoday.co.kr": "div#font", #34 아시아투데이
    "kado.net": "article.article-veiw-body", #35 강원도민일보
    "mbn.co.kr": "div#newsViewArea", #36 MBN
    "ksilbo.co.kr": "article.article-veiw-body", #37 경상일보
    "joongboo.com": "article.article-veiw-body", #38 중부일보
    "jbnews.com": "article.article-veiw-body", #39 중부매일
    "kwangju.co.kr": "div#joinskmbox", #40 광주일보
    "kwnews.co.kr": "div#articlebody", #41 강원일보
    "economist.co.kr": "div#article_body", #42 이코노미스트
    "sports.khan.co.kr": "div#articleBody",#43 스포츠경향
    "kgnews.co.kr": "div#news_body_area", #44 경기신문
    "nongmin.com": "div.news_txt.ck-content", #45 농민신문
    "yeongnam.com": "article.article-news-box", #46 영남일보
    "sisain.co.kr": "article.article-veiw-body", #47 시사IN
    "isplus.com": "div#article_body", #48 일간스포츠
    "inews365.com": "div.article", #49 충북일보
    "daejonilbo.com": "article.article-veiw-body", #50 대전일보
    "kihoilbo.co.kr": "article.article-veiw-body", #51 기호일보
    "newspenguin.com": "article.article-veiw-body", #52 뉴스펭귄
    "mediatoday.co.kr": "article.article-veiw-body", #53 미디어오늘
    "mdilbo.com": "div.article_view", #54 무등일보
    "kyeongin.com": "div#article-body", #55 경인일보
    "gnnews.co.kr": "div.news_text", #56 경남일보
    "sportsseoul.com": "div#article-body", #57 스포츠서울
    "idaegu.co.kr": "div.news_text", #58 대구신문
    "idaegu.com": "article.article-veiw-body", #59 대구일보
    "idomin.com": "article.article-veiw-body", #60 경남도민일보
    "namdonews.com": "article.article-veiw-body", #61 남도일보
    "obsnews.co.kr": "article.article-veiw-body", #62 OBS
    "kyongbuk.co.kr": "article.article-veiw-body", #63 경북일보
    "knnews.co.kr": "div.cont_cont", #64 경남신문
    "sports.hankooki.com": "article.article-veiw-body", #65 스포츠한국
    "jjan.kr": "div.article_txt_container", #66 전북일보
    "joongdo.co.kr": "div#font", #67 중도일보
    "hidomin.com": "div#article-view-content-div", #68 경북도민일보
    "naeil.com": "div.article-view", #69 내일신문
    "kjdaily.com": "div#content", #70 광주매일신문
    "cctoday.co.kr": "article.article-veiw-body", #71 충청투데이
    "jnilbo.com": "div#content", #72 전남일보
    "viva100.com": "div.news_content", #73 브릿지경제
    "sportsworldi.com": "article.viewBox2", #74 스포츠월드
    "sjbnews.com": "span.news_text.cl6.p-b-25", #75 새전북신문
    "dynews.co.kr": "article.article-veiw-body", #76 동양일보
    "iusm.co.kr": "article.article-veiw-body", #77 울산매일
    "dnews.co.kr": "div.text", #78 e대한경제
    "hellodd.com": "article.article-veiw-body", #79 헬로디디
    "ilyo.co.kr": "div.contentView.ctl-font-ty2.editorType2", #80 일요신문
    "ccdailynews.com": "article.article-veiw-body", #81 충청일보
    "djtimes.co.kr": "article.article-veiw-body", #82 당진시대
    "hkbs.co.kr": "article.article-veiw-body", #83 환경일보
    "h21.hani.co.kr": "div.arti-txt.0", #84 한겨레21
    "ihalla.com": "div.article_txt", #85 한라일보
    "ulsanpress.net": "article.article-veiw-body", #86 울산신문
    "jejunews.com": "div#article-view-content-div", #87 제주일보
    "wonjutoday.co.kr": "article.article-veiw-body", #88 원주투데이
    "kbmaeil.com": "div.news_content", #89 경북매일신문
    "weekly.hankooki.com": "article.article-veiw-body", #90 주간한국
    "yjinews.com": "article.article-veiw-body", #91 영주시민신문
    "ebn.co.kr": "article.article-veiw-body", #92 EBN산업뉴스
    "kidshankook.kr": "article.article-veiw-body", #93 소년한국일보
    "journalist.or.kr": "div#news_body_area", #94 기자협회보
    "jeollailbo.com": "article.article-veiw-body", #95 전라일보
    #"jemin.com": "article.article-veiw-body", #96 제민일보
    "kukinews.com": "div#articleContent", #97 쿠키뉴스
    "ekn.kr": "div#news_body_area_contents", #98 에너지경제
    "pttimes.com": "article.article-veiw-body", #99 평택시민신문
    "mediapen.com": "div#articleBody", #100미디어펜
    "koreatimes.com": "div#print_arti", #101코리아타임스
    "okinews.com": "div#article-view-content-div", #102옥천신문
    "igimpo.com": "article.article-veiw-body", #103김포신문
    #"gwangnam.co.kr": "div#content", #104광남일보
    "pdjournal.com": "article.article-veiw-body", #105PD저널
    "pennmike.com": "article.article-veiw-body", #106펜앤드마이크
    "hsnews.co.kr": "article.article-veiw-body", #107홍성신문
    "metroseoul.co.kr": "div.col-12", #108메트로경제
    "pressian.com": "div.article_body", #109프레시안
    "womaneconomy.co.kr": "article.article-veiw-body", #110여성경제신문
    #"wooriy.com": "", #111영암우리신문
    "gynet.co.kr": "div#article-view-content-div", #112광양신문
    "newssc.co.kr": "div#article-view-content-div", #113뉴스서천
    "kidkangwon.co.kr": "div#article-view-content-div", #114어린이강원
    "mygoyang.com": "article.article-veiw-body", #115주간고양신문
    "soraknews.co.kr": "td#ct", #116주간설악신문
    "seoulwire.com": "article.article-veiw-body", #117서울와이어

    "news.mtn.co.kr": "div.css-x1j506"
}

def _normalize_domain(netloc: str) -> str:
    d = netloc.lower()
    return d[4:] if d.startswith("www.") else d

def _pick_selector(netloc: str, selector_map: dict) -> str | None:
    """정확 매치 → www 제거 매치 → endswith 매치 순으로 셀렉터 선택"""
    if netloc in selector_map:
        return selector_map[netloc]
    nd = _normalize_domain(netloc)
    if nd in selector_map:
        return selector_map[nd]
    for key in selector_map:
        if nd.endswith(key):
            return selector_map[key]
    return None

def fallback_with_requests(url: str) -> str:
    """
    기본: requests + BeautifulSoup로 본문 추출
    - 도메인별 selector_map 우선
    - 실패 시 <p> 태그 텍스트 모아 반환
    - 줄바꿈은 유지하여 추후 비교에 유리하게 처리
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code != 200:
            return ""
        # 인코딩 핸들링(일부 언론사 EUC-KR 등)
        if "kookje.co.kr" in url:
            res.encoding = "euc-kr"
        elif res.apparent_encoding:
            res.encoding = res.apparent_encoding

        try:
            soup = BeautifulSoup(res.text, "html.parser")
        except Exception:
            soup = BeautifulSoup(res.content, "html.parser")

        selector = _pick_selector(urlparse(url).netloc, selector_map)
        if selector:
            content = soup.select_one(selector)
            if content:
                for tag in content.select("script, style, iframe"):
                    tag.decompose()
                return content.get_text(separator="\n", strip=True)

        # fallback: 모든 <p> 합치기
        return "\n".join(
            p.get_text(separator="\n", strip=True) for p in soup.find_all("p")
        )
    except Exception as e:
        log(f"⚠️ fallback 요청 예외: {e} - url: {url}")
        return ""

# =========================
# 유사도 계산
# =========================

def calculate_copy_ratio(article: str, post: str) -> float:
    """TF-IDF + 코사인 유사도 기반 (문장 단위 평균)"""
    def _clean(t): return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", t or "")).strip()
    article, post = _clean(article), _clean(post)
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', article) if s.strip()]
    if not sentences:
        return 0.0
    scores = []
    for s in sentences:
        try:
            v = TfidfVectorizer(tokenizer=okt.morphs).fit([s, post])
            tfidf = v.transform([s, post])
            scores.append(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])
        except Exception:
            continue
    return round(sum(scores) / len(scores), 3) if scores else 0.0

def calculate_sequence_matcher_ratio(article: str, post: str) -> float:
    """
    SequenceMatcher 기반 단방향 복사율:
      - article(원문) 중 post(블로그)에 포함되는 비율
    """
    def _clean(t):
        t = "" if t is None else str(t)
        t = re.sub(r"[^\w\s]", "", t)
        t = re.sub(r"\s+", " ", t)
        return t.strip()

    article_clean = _clean(article)
    post_clean = _clean(post)
    if not article_clean or not post_clean:
        return 0.0

    matcher = SequenceMatcher(None, post_clean, article_clean)
    matched_len = sum(b.size for b in matcher.get_matching_blocks() if b.size > 0)
    return round(matched_len / len(article_clean), 3)

# === 문장 완전/거의-일치 기반 복제율 ===
from html import unescape
import difflib

# 선택 의존성: kss가 있으면 더 좋은 문장 분리, 없으면 정규식 fallback
try:
    import kss
    _HAS_KSS_EXACT = True
except Exception:
    _HAS_KSS_EXACT = False

def _normalize_for_exact(s: str) -> str:
    """HTML/구두점/공백 정규화: 완전/거의-일치 판정의 전처리."""
    s = "" if s is None else str(s)
    s = unescape(s)
    s = re.sub(r'<[^>]+>', ' ', s)  # 태그 제거
    s = s.replace('“','"').replace('”','"').replace('’',"'").replace('‘',"'")
    s = s.replace('–','-').replace('—','-')
    s = s.replace('（','(').replace('）',')').replace('[','(').replace(']',')')
    s = re.sub(r'(?<=\d),(?=\d)', '', s)  # 숫자 콤마 제거
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def _split_sentences_exact(text: str) -> list[str]:
    """문장 분리(kss 우선, 미설치 시 마침표/물음표/느낌표/줄바꿈 기준)."""
    text = "" if text is None else str(text).strip()
    if not text:
        return []
    if _HAS_KSS_EXACT:
        return [s.strip() for s in kss.split_sentences(text) if s and s.strip()]
    parts = re.split(r'(?<=[.!?])\s+|\n+', text)
    return [p.strip() for p in parts if p and p.strip()]

def _is_valid_sentence_exact(s: str, min_chars=20, min_tokens=5) -> bool:
    """너무 짧거나 토큰 적은 문장은 제외(잡음 완화)."""
    if len(s) < min_chars:
        return False
    if len(re.findall(r'\w+', s)) < min_tokens:
        return False
    return True

def _almost_equal_exact(a: str, b: str, tol: float = 0.98) -> bool:
    """공백/구두점 등 미세차이를 허용하는 '거의-일치' 판정."""
    return difflib.SequenceMatcher(a=a, b=b, autojunk=False).ratio() >= tol

def calculate_exact_copy_rate(article_text: str,
                              post_text: str,
                              mode: str = "hybrid",      # "sentence" | "substr" | "hybrid"
                              min_chars: int = 20,
                              min_tokens: int = 5,
                              almost_tol: float = 0.98) -> float:
    """
    정확 복제율 = (일치(또는 거의-일치) 문장 수) / (원문 유효 문장 수)
    반환: 0.00 ~ 1.00 (소수 둘째 자리 반올림)
      - mode="sentence": 문장→문장 완전/거의-일치 집합 매칭
      - mode="substr":   원문 문장이 게시글 전체 텍스트의 서브스트링인지 확인
      - mode="hybrid":   sentence 우선, 남은 문장만 substr로 보완
    """
    # 1) 문장 분리 + 정규화
    A = [_normalize_for_exact(x) for x in _split_sentences_exact(article_text)]
    A = [x for x in A if _is_valid_sentence_exact(x, min_chars, min_tokens)]
    if not A:
        return 0.0

    P_sent = [_normalize_for_exact(x) for x in _split_sentences_exact(post_text)]
    P_set  = set(P_sent)
    P_all  = _normalize_for_exact(post_text)

    copied = 0

    if mode in ("sentence", "hybrid"):
        # 1차: 완전일치
        unmatched = []
        for s in A:
            if s in P_set:
                copied += 1
            else:
                unmatched.append(s)

        # 2차: 거의-일치(길이 20% 이내 후보만 비교)
        if unmatched:
            for s in list(unmatched):
                cands = [t for t in P_sent if abs(len(t) - len(s)) / max(1, len(s)) < 0.2]
                if any(_almost_equal_exact(s, t, tol=almost_tol) for t in cands):
                    copied += 1
                    unmatched.remove(s)

        # 3차: hybrid면 남은 문장 substr 확인
        if mode == "hybrid" and unmatched:
            for s in unmatched:
                if s and s in P_all:
                    copied += 1

    elif mode == "substr":
        for s in A:
            if s and s in P_all:
                copied += 1

    total = len(A)
    return round(copied / total, 2) if total else 0.0


# =========================
# 네이버 뉴스 API 검색
# =========================

def search_naver_news_api(queries: list[str], index: int,
                          client_id: str, client_secret: str) -> list[dict]:
    """
    쿼리별 네이버 뉴스 API → 후보 링크 수집 → 본문 크롤링 → 결과 리스트 반환
    결과 원소: {"title": str, "link": str, "body": str}
    """
    # 도메인 화이트리스트 로드
    trusted_domains_file = find_resource("매체사_도메인_정보.xlsx")
    trusted_domains = _safe_read_excel(
        trusted_domains_file, required_col="도메인", default_list=[]
    )

    headers = {"X-Naver-Client-Id": client_id, "X-Naver-Client-Secret": client_secret}
    results: list[dict] = []
    seen_links: set[str] = set()

    for q in queries:
        try:
            url = f"https://openapi.naver.com/v1/search/news.json?query={urllib.parse.quote(q)}&display=15&sort=sim"
            res = requests.get(url, headers=headers, timeout=10)
            time.sleep(0.25)  # 레이트 리밋 완화

            if res.status_code != 200:
                log(f"❌ API 응답 오류 [{res.status_code}] - query: {q}", index)
                log(f"↪ 응답 내용: {res.text[:300]}...", index)
                continue

            try:
                data = res.json()
            except Exception as e:
                log(f"❌ JSON 파싱 실패: {e} - query: {q}", index)
                log(f"↪ 원본 응답: {res.text[:300]}...", index)
                continue

            for item in data.get("items", []):
                link = item.get("link")
                title = item.get("title")
                if not link or link in seen_links or is_excluded(link):
                    continue

                # 네이버 도메인 OID 필터(신탁 언론만)
                if "naver.com" in link:
                    oid = extract_oid_from_naver_url(link)
                    if not oid:
                        log(f"⚠️ OID 추출 실패 → 스킵: {link}", index)
                        continue
                    if "n.news.naver.com" in link and oid not in trusted_news_oids:
                        continue
                    if "sports.naver.com" in link and oid not in trusted_sports_oids:
                        continue
                    if "entertain.naver.com" in link and oid not in trusted_entertain_oids:
                        continue

                # 도메인 화이트리스트 필터 추가
                else:
                    netloc = urlparse(link).netloc.lower()
                    if not any(netloc.endswith(d) for d in trusted_domains):
                        log(f"🚫 비신탁 도메인 제외 : {link}", index)
                        continue

                seen_links.add(link)
                body = fallback_with_requests(link)
                if body and len(body) > 300:
                    results.append({
                        "title": title,
                        "link": link,
                        "body": clean_text(body, preserve_newline=True),
                        "query": q,
                    })

        except Exception as e:
            log(f"❌ API 요청 예외: {e} - query: {q}", index)

    return results
