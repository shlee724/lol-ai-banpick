from core.window_tracker import WindowTracker
from core.screen_capture import capture_window
from core.roi_manager import crop_roi_definite_xy, crop_roi_relative_xy
from config.roi import ROI
from config.path import PATHS
from pipeline.normalizer import TextNormalizer
from pipeline.classifier import StateClassifier
from pipeline.buffer import StateBuffer
from pipeline.state_manager import StableStateManager
from pipeline.pick_stage_detector import detect_pick_kind_from_banned_strips
from core.ocr_engine import extract_text
from core.gemini_vision import analyze_image_json
from config.prompts import PICKED_CHAMPS_WITH_ROLES_PROMPT, BANNED_CHAMPS_10_PROMPT, DRAFT_FROM_IMAGE_PROMPT_LITE
from config.prompts import build_draft_recommend_prompt, build_draft_recommend_prompt_lite
from core.draft_schema import normalize_picks_with_roles, normalize_bans10, safe_get_draft_fields
from core.gemini_text import generate_text_json
from core.lol_pick_coach import lol_mid_pick_coach_stream, get_client
from core.lol_playplan_coach import lol_playplan_stream, get_playplan_coach_client
from pipeline.prepare_phase_detector import is_dual_timer_effective
from PIL import Image
import time
import json

tracker = WindowTracker("League of Legends")
normalizer = TextNormalizer()
classifier = StateClassifier()
buffer = StateBuffer(size=7)
state_manager = StableStateManager(
    min_duration=1.0,
    min_confidence=0.7
)
pick_coach_client = get_client()
playplan_coach_client = get_playplan_coach_client()

SLEEP_SEC = 0.01
PICK_STD_THRESHOLD = 30.0
MODEL_VISION = "gemini-3-flash-preview"
MODEL_TEXT = "gemini-3-flash-preview"

MY_ROLE = "MID"   # TOP/JUNGLE/MID/ADC/SUPPORT 중 하나로 고정
MY_TIER = "BRONZE"     # UNRANKED/IRON/BRONZE/SILVER/GOLD/PLATINUM/EMERALD/DIAMOND/MASTER/GRANDMASTER/CHALLENGER
MY_CHAMP_POOL = ["Malzahar", "Oriana", "Galio", "Mundo", "Garen", "Malphite", "Cho'gath", "Nasus", "kassadin"]  # 예시

DEBUG_SAVE = False
pick_real_executed = False  # PICK_REAL 알고리즘 1회 실행 보장

def merge_images_horizontal(img1: Image.Image, img2: Image.Image, bg_color=(255, 255, 255)) -> Image.Image:
    new_width = img1.width + img2.width
    new_height = max(img1.height, img2.height)

    new_img = Image.new("RGB", (new_width, new_height), bg_color)
    new_img.paste(img1, (0, 0))
    new_img.paste(img2, (img1.width, 0))
    return new_img

while True:
    window_rect_screen = tracker.get_window_rect()
    if window_rect_screen is None:
        print("롤 클라이언트 찾을 수 없음")
    elif window_rect_screen and tracker.hwnd:
        x, y, w, h = window_rect_screen
        window_size = (w, h)
        #print(f"창 위치: ({x},{y}) 크기: {w}x{h}")
        frame_img = capture_window(tracker.hwnd, w, h)        #롤 클라이언트 전체 이미지 (Image.Image)
        status_img = crop_roi_relative_xy(frame_img, window_size ,ROI.BANPICK_STATUS_TEXT)   #밴픽 상태메시지 캡처

        if DEBUG_SAVE:
            frame_img.save(PATHS.LOL_CLIENT_CAPTURE_PNG)
            status_img.save(PATHS.BANPICK_STATUS_TEXT_CAPTURE_PNG)

        my_banned_img = crop_roi_relative_xy(frame_img, window_size, ROI.BANNED_CHAMPIONS_MY_TEAM)
        enemy_banned_img = crop_roi_relative_xy(frame_img, window_size, ROI.BANNED_CHAMPIONS_ENEMY_TEAM)
        my_picked_img = crop_roi_relative_xy(frame_img, window_size, ROI.PICKED_CHAMPIONS_MY_TEAM)
        enemy_picked_img = crop_roi_relative_xy(frame_img, window_size, ROI.PICKED_CHAMPIONS_ENEMY_TEAM)
        bans_merged_img = merge_images_horizontal(my_banned_img, enemy_banned_img)
        picks_merged_img = merge_images_horizontal(my_picked_img, enemy_picked_img)
        banpick_timer_bar_img = crop_roi_relative_xy(frame_img, window_size, ROI.BANPICK_TIMER_BAR)
        banpick_timer_digit_img = crop_roi_relative_xy(frame_img, window_size, ROI.BANPICK_TIMER_DIGITS)
        
        # OCR
        status_text_raw = extract_text(status_img)

        # Pipeline
        status_text_norm = normalizer.normalize(status_text_raw)
        raw_state = classifier.classify(status_text_norm)

        buffer.push(raw_state)
        major_state = buffer.get_majority()
        major_conf = buffer.get_confidence()

        stable_state = state_manager.update(major_state, major_conf)       
        #print(f" StableState → {stable_state}") 



        if stable_state == "PICK":

            if pick_real_executed:
                #print(" (PICK_REAL algo already executed once - skip)")
                continue

            pick_res = detect_pick_kind_from_banned_strips(my_banned_img, enemy_banned_img, std_threshold=PICK_STD_THRESHOLD)
            print("PICK 판정:", pick_res.kind, "std:", round(pick_res.std, 2))

            if pick_res.kind == "PICK_REAL":
                # 진짜 픽 단계 로직 실행
                # 제미나이 api에 픽 정보 보내기
                try:
                    buf = []
                    stream_start_t = time.perf_counter()
                    first_token_t = None

                    for delta in lol_mid_pick_coach_stream(picks_merged_img, client=pick_coach_client, model="gemini-2.5-pro"):
                        if first_token_t is None:
                            first_token_t = time.perf_counter()
                            print(f"\n⏱ 첫 토큰: {first_token_t - stream_start_t:.2f}s\n")

                        print(delta, end="", flush=True)
                        buf.append(delta)

                    stream_end_t = time.perf_counter()
                    print(f"\n\n⏱ 전체: {stream_end_t - stream_start_t:.2f}s")
                    final_text = "".join(buf)

                except Exception as e:
                    print(" ❌ Gemini 호출 실패:", repr(e))
                    continue                

                pick_real_executed = True
                continue

        if stable_state == "PREPARE":
            dual_now = is_dual_timer_effective(
                timer_bar_img=banpick_timer_bar_img,
                timer_digits_img=banpick_timer_digit_img,
            )

            dual_buf.push(dual_now)
            dual_stable = dual_buf.get_majority()
            dual_conf = dual_buf.get_confidence()

            print(f" DualEffective → now={dual_now} stable={dual_stable} ({dual_conf:.2f})")

            # ✅ 확정 조건: 다수결 True + 신뢰도 임계(선택)
            if dual_stable is True and dual_conf >= 0.72:
                print("양팀 모든 챔피언 픽 됐습니다 (stable)")
                
                buf = []
                stream_start_t = time.perf_counter()
                first_token_t = None

                for delta in lol_playplan_stream(
                    picks_merged_img,
                    client=playplan_coach_client,
                    model="gemini-2.5-pro",
                ):
                    if first_token_t is None:
                        first_token_t = time.perf_counter()
                        print(f"\n⏱ 첫 토큰: {first_token_t - stream_start_t:.2f}s\n")

                    print(delta, end="", flush=True)
                    buf.append(delta)

                stream_end_t = time.perf_counter()
                print(f"\n\n⏱ 전체: {stream_end_t - stream_start_t:.2f}s")

                final_text = "".join(buf)
                break
        else:
            dual_buf = StateBuffer(size=7)  

    time.sleep(SLEEP_SEC)
