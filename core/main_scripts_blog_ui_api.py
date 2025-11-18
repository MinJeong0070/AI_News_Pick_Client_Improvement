import os
import re
import pandas as pd
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from core.core_utils_ui_api import (
    clean_text, extract_first_sentences, generate_search_queries,
    search_naver_news_api, calculate_copy_ratio, log, calculate_sequence_matcher_ratio,
    calculate_exact_copy_rate, load_trusted_domains, extract_urls_from_text, is_trusted_url, evaluate_single_article_url,
    is_whitelisted_domain, is_trusted_oid, fallback_with_requests,
)
import sys

def resource_path(relative_path):
    """兼容PyInstaller和源码运行的资源路径"""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

def find_original_article_api(
        index: int,
        row_dict: dict,
        output_dir: str,
        stop_event_flag: bool,
        client_id: str,
        client_secret: str,
        driver=None,
):
    try:
        title = (row_dict.get("게시글제목") or "").strip()
        content = (row_dict.get("게시글내용") or "").strip()
        if not title and not content:
            log("⚠️ 제목/본문 비어있음 → 스킵", index)
            return index, "", 0.0, "", 0.0, 0.0, False

        # -----------------------------
        # (A) 블로그 본문 내 URL 1차 필터 (요청한 2단계)
        # -----------------------------
        inpost_urls = extract_urls_from_text(content)
        valid_inpost_urls = [u for u in inpost_urls if is_whitelisted_domain(u) or is_trusted_oid(u)]

        # 블로그 본문에 URL이 있었는데 허용 URL이 하나도 없으면 → 행 삭제
        if inpost_urls and not valid_inpost_urls:
            log("🚫 블로그 내 URL 존재하지만 화이트리스트/신탁 OID 전처리 → 행 삭제", index)
            return index, "", 0.0, "", 0.0, 0.0, True  # ← delete_row_flag=True

        # 블로그 후보(본문 URL 통과분) 점수 선계산
        # 블로그 URL 후보 (본문 URL 통과분) 점수 선계산
        blog_candidate = None
        if valid_inpost_urls:
            blog_url = valid_inpost_urls[0]
            body = fallback_with_requests(blog_url)  # ← 여기만 사용
            if body and len(body) > 100:
                seq = calculate_sequence_matcher_ratio(clean_text(body), clean_text(content))
                exact = calculate_exact_copy_rate(clean_text(body), clean_text(content))
                blog_candidate = {"link": blog_url, "body": body, "seq": seq, "exact": exact, "tfidf": 0.0,
                                  "source": "inpost"}
                log(f"🧷 블로그 URL 후보: {blog_url} (Seq={seq:.3f}, Exact={exact:.3f})", index)

        # -----------------------------
        # (B) 네이버 뉴스 후보 수집
        # -----------------------------
        title = (row_dict.get("게시글제목") or "").strip()
        content = (row_dict.get("게시글내용") or "").strip()
        press = (row_dict.get("검색어") or "").strip()  # ← 언론사/매체사 표준 컬럼

        first, second, last = extract_first_sentences(content)

        queries = generate_search_queries(
            title, first, second, last, press, full_content=content
        )
        candidates = search_naver_news_api(queries=queries, index=index, client_id=client_id, client_secret=client_secret)

        # 후보 본문 불러오기 + 시퀀스 점수(요청한 4단계: 필터 전에 계산)
        enriched = []
        for it in candidates:
            link = it.get("link", "")
            body = it.get("body", "")
            if not body:
                body = fallback_with_requests(link)
            if not body or len(body) < 100:
                continue
            seq = calculate_sequence_matcher_ratio(clean_text(body), clean_text(content))
            exact = calculate_exact_copy_rate(clean_text(body), clean_text(content))
            tfidf = calculate_copy_ratio(clean_text(body), clean_text(content))
            enriched.append({
                "link": link,
                "body": body,
                "seq": seq,
                "exact": exact,
                "tfidf": tfidf,  # 필요 시 기존 TF-IDF 점수도 병기 가능
                "source": "api",
                "query": it.get("query", "")
            })

        if not enriched and not blog_candidate:
            log("❌ 후보 기사 없음", index)
            return index, "", 0.0, "", 0.0, 0.0, False

        # -----------------------------
        # (C) 후보 전체 중 1위(시퀀스 기준) 산출
        # -----------------------------
        pool = enriched.copy()
        if blog_candidate:
            pool.append(blog_candidate)
        pool.sort(key=lambda x: (x["seq"], x["exact"]), reverse=True)
        top_overall = pool[0]  # 시퀀스 1위

        # -----------------------------
        # (D) 1위가 비공식 매체면 → 행 삭제 (요청한 5단계 강화)
        #     (신탁 OID X & 화이트리스트 X)
        # -----------------------------
        if not (is_trusted_oid(top_overall["link"]) or is_whitelisted_domain(top_overall["link"])):
            log(f"🚫 1위가 비공식 매체 → 블로그 행 삭제: {top_overall['link']}", index)
            return index, "", 0.0, "", 0.0, 0.0, True  # ← delete_row_flag=True

        # -----------------------------
        # (E) 공식 매체만 남기고 최종 비교
        #     - 블로그 후보가 있으면 같이 비교
        #     - 동일 점수면 Seq 우선, 동률이면 Exact로 타이브레이크
        # -----------------------------
        trusted_only = [x for x in enriched if (is_trusted_oid(x["link"]) or is_whitelisted_domain(x["link"]))]
        if blog_candidate:
            trusted_only.append(blog_candidate)
        if not trusted_only:
            log("⚠️ 공식 매체 후보 없음", index)
            return index, "", 0.0, "", 0.0, 0.0, False

        trusted_only.sort(key=lambda x: (x["seq"], x["exact"]), reverse=True)
        best = trusted_only[0]

        # ✅ 원문 기사 텍스트 저장 (검색어 기반 파일명)
        try:
            used_query = best.get("query", "검색어 없음")
            safe_query = re.sub(r'[\\/*?:"<>|]', '', used_query).strip()[:100] or "검색어 없음"
            filename = os.path.join(output_dir, f"{index + 1:03d}_{safe_query}.txt")

            body_with_newline = best["body"].replace('\n', '\n')
            with open(filename, "w", encoding="utf-8", errors="replace") as f:
                f.write(f"[검색어] {used_query}\n[URL] {best['link']}\n\n{body_with_newline}")

            log(f"📝 저장 완료 → {filename} "
                f"(Seq:{best['seq']:.3f}, Exact:{best['exact']:.3f}, TF-IDF:{best.get('tfidf', 0.0):.3f})", index)
        except Exception as e:
            log(f"⚠️ 기사 저장 실패: {e}", index)

        # 최종 리턴
        return (
            index,
            best["link"],
            best.get("tfidf", 0.0),
            best["body"],
            float(best["seq"]),
            float(best["exact"]),
            False  # delete_row_flag
        )

    except Exception as e:
        log(f"❌ 결과 처리 오류: {e}", index)
        return index, "", 0.0, "", 0.0, 0.0, False

def clean_surrogates(val):
    """非法 surrogate 제거"""
    if isinstance(val, str):
        # U+D800 - U+DFFF 范围字符去掉
        return re.sub(r'[\ud800-\udfff]', '', val)
    return val

def main(input_path, output_path, client_id, client_secret, stop_event=None):
    output_dir = os.path.splitext(output_path)[0] + "_본문"
    os.makedirs(output_dir, exist_ok=True)

    # 0) 가장 큰 시트를 먼저 확정
    def _load_best_sheet(path):
        xls = pd.ExcelFile(path)
        best_df, best_len = None, -1
        for s in xls.sheet_names:
            tmp = pd.read_excel(xls, sheet_name=s, dtype=str, keep_default_na=False, engine="openpyxl")
            tmp = tmp.applymap(lambda x: str(x).strip().replace("\u200b", "") if pd.notna(x) else "")
            if len(tmp) > best_len:
                best_df, best_len = tmp, len(tmp)
        return best_df

    df = _load_best_sheet(input_path)  # ← 먼저 확정

    # 1) 컬럼 표준화
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

    # 2) 결과 컬럼 초기화(없으면 추가)
    for col, val in [
        ("원문기사 URL", ""),
        ("원문내용", ""),
        ("TF-IDF", 0.0),
        ("SequenceMatcher유사도", 0.0),
        ("정확복제율", 0.0),
    ]:
        if col not in df.columns:
            df[col] = val

    # 3) 로그 및 총 건수
    total = len(df)
    log(f"📄 전체 게시글 수: {total}")
    log(f"📑 컬럼: {list(df.columns)}")
    log(f"🔎 제목/내용 Not-blank: {((df['게시글제목'] != '') & (df['게시글내용'] != '')).sum()}행")

    def get_stop_flag():
        return stop_event.is_set() if stop_event else False

    tasks = [(i, row.to_dict(), output_dir, get_stop_flag(), client_id, client_secret)
             for i, row in df.iterrows()]

    with ProcessPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(find_original_article_api, *args) for args in tasks]
        try:
            for future in as_completed(futures):
                if stop_event and stop_event.is_set():
                    log("🛑 사용자 중단 요청 감지, 작업 중단")
                    executor.shutdown(cancel_futures=True)
                    break
                try:
                    index, link, tfidf_score, body, sequence_score, exact, delete_row_flag = future.result()

                    if delete_row_flag:
                        if 0 <= index < len(df):
                            df.drop(index, inplace=True)
                        continue

                    df.at[index, "원문기사 URL"] = link
                    df.at[index, "원문내용"] = body  # 문자열(기사 본문)
                    df.at[index, "TF-IDF"] = tfidf_score  # 숫자(TF-IDF 유사도)
                    df.at[index, "SequenceMatcher유사도"] = sequence_score
                    df.at[index, "정확복제율"] = exact
                except Exception as e:
                    log(f"❌ 결과 처리 오류: {e}")
                    continue

        except Exception as e:
            log(f"❌ 프로세스 풀 에러: {e}")

    for c in ["TF-IDF", "SequenceMatcher유사도", "정확복제율"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    matched_count = df["TF-IDF"].gt(0).sum()
    above_90_count = df["TF-IDF"].ge(0.9).sum()
    above_50_count = df["TF-IDF"].ge(0.5).sum() - above_90_count
    above_0_count = matched_count - above_90_count - above_50_count

    stats_rows = pd.DataFrame([
        {"순번": "매칭건수", "검색": f"{matched_count}건"},
        {"순번": "0.5 이상", "검색": f"{above_50_count}건"},
        {"순번": "0.9 이상", "검색": f"{above_90_count}건"},
        {"순번": "0 이상", "검색": f"{above_0_count}건"},
    ])

    # '순번' 컬럼 제거
    if "순번" in df.columns:
        df.drop(columns=["순번"], inplace=True, errors="ignore")

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
