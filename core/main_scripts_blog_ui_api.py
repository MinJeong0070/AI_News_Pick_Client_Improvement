import os
import re
import pandas as pd
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from core.core_utils_ui_api import (
    clean_text, extract_first_sentences, generate_search_queries,
    search_naver_news_api, calculate_copy_ratio, log, calculate_sequence_matcher_ratio
, calculate_exact_copy_rate
)
import sys

def resource_path(relative_path):
    """兼容PyInstaller和源码运行的资源路径"""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

def find_original_article_api(index, row_dict, total_count, output_dir, stop_event_flag, client_id, client_secret):
    try:
        # 检查中断
        if stop_event_flag:
            log("🛑 사용자 중단 요청 감지, 작업 중단", index)
            return index, "", 0.0, "", 0.0, 0.0

        title = clean_text(str(row_dict.get("게시글제목", "")))
        content = clean_text(str(row_dict.get("게시글내용", "")))
        press = clean_text(str(row_dict.get("검색어", "")))
        first, second, last = extract_first_sentences(content)
        queries = generate_search_queries(title, first, second, last, press)
        log(f"🔍 검색어: {queries}", index)

        # 检查中断
        if stop_event_flag:
            log("🛑 사용자 중단 요청 감지, 작업 중단", index)
            return index, "", 0.0, "", 0.0, 0.0

        search_results = search_naver_news_api(queries, index, client_id, client_secret)
        if not search_results:
            log("❌ 관련 뉴스 없음", index)
            return index, "", 0.0, "", 0.0, 0.0

        if stop_event_flag:
            log("🛑 사용자 중단 요청 감지, 작업 중단", index)
            return index, "", 0.0, "", 0.0, 0.0
        '''
        # 기존 TF-IDF 방식
        best = max(search_results, key=lambda x: calculate_copy_ratio(x["body"], title + " " + content))
        score = calculate_copy_ratio(best["body"], title + " " + content)
        sequence_score = calculate_sequence_matcher_ratio(best["body"], content)
        exact = calculate_exact_copy_rate(best["body"], title + " " + content, mode="hybrid")  # ✅ 정확복제율
        '''
        best = max(search_results, key=lambda x: calculate_sequence_matcher_ratio(x["body"], title + " " + content))
        score = calculate_sequence_matcher_ratio(best["body"], title + " " + content)
        tfidf_score = calculate_copy_ratio(best["body"], title + " " + content)
        exact = calculate_exact_copy_rate(best["body"], title + " " + content, mode="hybrid")

        # 保留换行的正文
        body_with_newline = best["body"]

        if score >= 0.0:
            safe_title = re.sub(r'[\\/*?:"<>|]', '', title)[:50]
            filename = os.path.join(output_dir, f"{index + 1:03d}_{safe_title}.txt")
            with open(filename, "w", encoding="utf-8", errors="replace") as f:
                f.write(f"[URL] {best['link']}\n\n{body_with_newline}")
            log(f"📝 저장 완료 → {filename} (복사율: {score})", index)
            hyperlink = f'=HYPERLINK("{best["link"]}")'
            return index, hyperlink, tfidf_score, body_with_newline, score, exact
        else:
            log(f"⚠️ 복사율 낮음 (복사율: {score})", index)
            return index, "", 0.0, "", 0.0, 0.0
    except Exception as e:
        log(f"❌ 에러 발생: {e}", index)
        return index, "", 0.0, "", 0.0, 0.0

def clean_surrogates(val):
    """非法 surrogate 제거"""
    if isinstance(val, str):
        # U+D800 - U+DFFF 范围字符去掉
        return re.sub(r'[\ud800-\udfff]', '', val)
    return val

def main(input_path, output_path, client_id, client_secret, stop_event=None):
    output_dir = os.path.splitext(output_path)[0] + "_본문"
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_excel(input_path, dtype={"게시글 등록일자": str})
    total = len(df)
    log(f"📄 전체 게시글 수: {total}개")

    df["원본기사"] = ""
    df["복사율"] = 0.0
    df["원문내용"] = ""  # 新增列
    df["SequenceMatcher유사도"] = 0.0  # 新增列

    # 가장 큰 시트를 자동 선택
    def _load_best_sheet(path):
        xls = pd.ExcelFile(path)
        best, n = None, -1
        for s in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=s, dtype=str, keep_default_na=False, engine="openpyxl")
            df = df.applymap(lambda x: str(x).strip().replace("\u200b", "") if pd.notna(x) else "")
            if len(df) > n:
                best, n = df, len(df)
        return best

    # 1️⃣ 가장 큰 시트 자동 선택
    df = _load_best_sheet(input_path)

    # 2️⃣ 컬럼 표준화
    rename_map = {
        "게시물 제목": "게시글제목",
        "게시물 내용": "게시글내용",
        "언론사": "검색어",
        "매체사": "검색어"
    }
    for a, b in rename_map.items():
        if a in df.columns and b not in df.columns:
            df[b] = df[a]
    for c in ["게시글제목", "게시글내용", "검색어"]:
        if c not in df.columns:
            df[c] = ""

    # 3️⃣ 불필요한 첫 로그 제거 후, 이 시점에 한 번만 로그 출력
    log(f"📄 전체 게시글 수: {len(df)}")
    log(f"📑 컬럼: {list(df.columns)}")
    log(f"🔎 제목/내용 Not-blank: {((df['게시글제목'] != '') & (df['게시글내용'] != '')).sum()}행")

    def get_stop_flag():
        return stop_event.is_set() if stop_event else False

    tasks = [(i, row.to_dict(), total, output_dir, get_stop_flag(), client_id, client_secret) for i, row in
             df.iterrows()]

    with ProcessPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(find_original_article_api, *args) for args in tasks]
        try:
            for future in as_completed(futures):
                if stop_event and stop_event.is_set():
                    log("🛑 사용자 중단 요청 감지, 작업 중단")
                    executor.shutdown(cancel_futures=True)
                    break
                try:
                    index, link, tfidf_score, body, sequence_score, exact = future.result()
                    df.at[index, "원본기사"] = link
                    df.at[index, "원문내용"] = body  # 문자열(기사 본문)
                    df.at[index, "복사율"] = tfidf_score  # 숫자(TF-IDF 유사도)
                    df.at[index, "SequenceMatcher유사도"] = sequence_score
                    df.at[index, "정확복제율"] = exact
                except Exception as e:
                    log(f"❌ 결과 처리 오류: {e}")

        except Exception as e:
            log(f"❌ 프로세스 풀 에러: {e}")

    matched_count = df["복사율"].gt(0).sum()
    above_90_count = df["복사율"].ge(0.9).sum()
    above_50_count = df["복사율"].ge(0.5).sum() - above_90_count
    above_0_count = matched_count - above_90_count - above_50_count

    stats_rows = pd.DataFrame([
        {"순번": "매칭건수", "검색": f"{matched_count}건"},
        {"순번": "0.5 이상", "검색": f"{above_50_count}건"},
        {"순번": "0.9 이상", "검색": f"{above_90_count}건"},
        {"순번": "0 이상", "검색": f"{above_0_count}건"},
    ])
    df = pd.concat([df, stats_rows], ignore_index=True)

    # surrogate 문자 제거
    df = df.applymap(clean_surrogates)

    df.to_excel(output_path, index=False)

    log("📊 통계 요약")
    log(f" 매칭건수: {matched_count}건")
    log(f" 0.5 이상: {above_50_count}건")
    log(f" 0.9 이상: {above_90_count}건")
    log(f" 0 이상: {above_0_count}건")
    log(f"🎉 완료! 저장됨 → {output_path}")
