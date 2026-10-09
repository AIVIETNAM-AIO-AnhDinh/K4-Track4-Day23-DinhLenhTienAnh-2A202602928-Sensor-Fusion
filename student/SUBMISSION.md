# Báo cáo bài nộp — Day 23 Sensor Fusion Lab

> Điền file này rồi commit. Cách nộp: [hướng dẫn nộp](../SUBMISSION.md).

## Thông tin học viên

- Họ tên: Dinh Lenh Tien Anh
- MSSV: 2A202602928
- Email:
- Link repo (fork): https://github.com/AIVIETNAM-AIO-AnhDinh/K4-Track4-Day23-DinhLenhTienAnh-2A202602928-Sensor-Fusion
- Commit hash nộp (`git rev-parse HEAD`): xem bài nộp trên LMS (commit cuối trên `main`)

## Tóm tắt kết quả

- `fusion_mode` (bắt buộc `compare`), `frames`, `segment`, `seed`: `compare`, `[0, 198]`, `training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord`, `0`
- `detection.precision`, `detection.recall`, `detection.tp/fp/fn`: 0.9701, 0.7004, 519 / 16 / 222
- `tracking.lidar.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`: 0.1503 m, 502, 11.344, 0, 239, 2.523
- `tracking.fused.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`: 0.1359 m, 502, 9.267, 0, 239, 2.523
- Giải thích khác biệt hai mode, đọc RMSE cùng số ghép và ghost/miss:
  hai mode có **cùng** `matches` (502), `ghost_track_frames` (0), `missed_gt_frames` (239)
  và `mean_confirmed_tracks`; so từng dòng `grade_run_lidar.log` với `grade_run_fused.log`
  thì `confirmed/matches/ghosts/misses` trùng nhau ở cả 199 frame. Nghĩa là camera không
  tạo, xoá hay xác nhận track nào — đúng thiết kế track-then-fuse. Khác biệt chỉ nằm ở độ
  chính xác vị trí: `sum_sq_err` giảm 11.34 → 9.27, RMSE giảm 0.150 → 0.136 m (−0.014 m);
  fused tốt hơn ở 137 frame, kém hơn ở 58 frame. Camera chủ yếu siết hướng ngang (góc
  phương vị) mà lidar đã đo khá tốt, nên mức cải thiện nhỏ. 0 ghost cho
  `precision_track = 502/502 = 1.0`; `coverage = 502/519 = 0.97`. Trong 239 lượt miss có
  222 lượt detector bỏ sót (`det_fn`), 17 lượt còn lại là trễ xác nhận: track sinh ở frame 0
  chỉ confirmed từ frame 4 (score 5/6 > 0.8), thấy ở 4 frame đầu `confirmed = 0`, `misses = 2`.
  Lưu ý: camera ở đây là tâm hộp 2D ground-truth cộng nhiễu, nên RMSE fused là cận lạc quan.

Chạy từ root repo:

```bash
fusion-run-lab --config student/config/paths.yaml --fusion compare --seed 0
```

`rmse = sqrt(sum_sq_err/matches)` trên vị trí 3D của confirmed tracks ghép
một-một với GT xe trong cửa sổ BEV, gate XY **2.0 m**; `null` nếu không có cặp.
Camera dùng tâm hộp 2D ground-truth FRONT có nhiễu seeded, **không** dùng camera
detector. Kết quả này không đo hiệu quả một perception system độc lập với GT.

`grade_run.log` là JSONL, mỗi `(mode,frame)` đúng một record với các trường:
`mode`, `frame`, `det_tp`, `det_fp`, `det_fn`, `valid_gt`, `confirmed`, `matches`,
`sum_sq_err`, `ghosts`, `misses`. Đảm bảo `matches+ghosts==confirmed` và
`matches+misses==valid_gt`; tổng/trung bình record phải khớp `metrics.json`.
File per-mode `metrics_lidar.json`, `metrics_fused.json`, `grade_run_lidar.log`,
`grade_run_fused.log` được giữ để đối chiếu.

## Giải thích ngắn (Parts E–H — tự viết)

1. Khác biệt đo lidar 3D và camera 2D trong EKF (`z`, `R`)?

   Lidar: `z = (x, y, z)` (m) là tâm hộp 3D, `R = diag(0.1², 0.1², 0.1²)`; `h(x)` tuyến
   tính (`p_s = R·p + t`) nên `H` là ma trận 3×6 hằng `[Rot | 0]`. Camera: `z = (u, v)`
   (pixel), `R = diag(5², 5²)` (`build_camera_measurement`); `h(x)` là phép chiếu pinhole
   `u = c_i − f_i·y_s/x_s`, `v = c_j − f_j·z_s/x_s` — phi tuyến, nên EKF dùng Jacobian
   2×6 tính tại trạng thái dự đoán. Camera không đo độ sâu, chỉ ràng buộc hướng nhìn; và
   chỉ chiếu được khi điểm hữu hạn và `x_s > 1e-6` (nếu không `ValueError`).

2. Vì sao cần gating Mahalanobis trước khi gán?

   `d² = γᵀ S⁻¹ γ` chuẩn hoá innovation theo độ bất định thật `S = H P Hᵀ + R`, nên so
   được giữa lidar (m) và camera (pixel) và giữa track mới (P lớn) với track ổn định.
   Ngưỡng `chi2.ppf(0.995, dim_meas)` (3 bậc tự do cho lidar, 2 cho camera) loại các cặp
   phi lý trước khi greedy chọn cặp nhỏ nhất, tránh việc một đo sai/clutter kéo track đi
   xa. Cặp ngoài FOV được loại trước cả bước này (`association_cost_matrix` gán `inf`),
   để không chiếu điểm sau lưng camera.

3. Pipeline là track-then-fuse hay fuse-then-track? Chỉ ra trên log `fusion-run-lab`.

   Track-then-fuse. Mỗi frame: predict một lần cho mọi track → `associate_and_update`
   với đo lidar → `associate_and_update` với đo camera (`run_lab.py`, vòng frame). Trên
   log: `grade_run_fused.log` và `grade_run_lidar.log` có `confirmed`, `matches`,
   `ghosts`, `misses` giống hệt nhau ở mọi frame (ví dụ frame 0–3 cả hai đều
   `confirmed: 0`, frame 4 đều `confirmed: 2`), chỉ `sum_sq_err` khác. Track được tạo và
   quản lý bởi lidar; camera chỉ tinh chỉnh trạng thái của track đã có.

4. Nếu camera lệch calibration, triệu chứng gì trên innovation/residual?

   Innovation camera `γ = z − h(x)` sẽ có **độ lệch hệ thống** (trung bình khác 0, cùng
   dấu qua nhiều frame và nhiều track) thay vì nhiễu trắng quanh 0 với phương sai ≈ `S`.
   `d²` trung bình tăng vượt số bậc tự do (2), nhiều cặp bị gate chi-square loại hơn,
   và các update camera được nhận sẽ kéo vị trí track lệch ngang khỏi đo lidar — lidar và
   camera "giằng co", RMSE fused xấu hơn lidar. Lệch xoay (yaw) gây sai số tăng theo
   khoảng cách; lệch tịnh tiến gây sai số lớn ở vật gần.

5. Vì sao `associate_and_update(..., sensor)` cần sensor tường minh ở frame rỗng?
   Giải thích vì sao lidar quyết định score/init/delete còn camera chỉ EKF update.

   Khi `meas_list` rỗng không thể suy ra loại cảm biến từ `meas.sensor`, nhưng frame lidar
   rỗng vẫn phải chạy `manage_tracks`: mọi track trong FOV lidar bị tính miss
   (`−1/window`) và có thể bị xoá; còn lượt camera rỗng thì không được làm gì với score.
   Vì vậy hàm luôn gọi `manager.manage_tracks(unassigned_tracks, unassigned_meas, sensor)`
   với sensor của lượt. Lidar đo trực tiếp vị trí 3D trong FOV 180° phía trước, đủ để
   khởi tạo trạng thái và đánh giá tồn tại; camera thiếu độ sâu, FOV hẹp và (trong lab)
   lấy từ nhãn GT, nên nếu cho camera cộng/trừ score thì một track sẽ được xác nhận hoặc
   bị xoá hai lần mỗi frame và phụ thuộc vào việc xe có nằm trong ảnh FRONT hay không.

6. Nêu điều kiện xác nhận, giữ confirmed sau miss, và điều kiện xóa track.

   Track mới: `score = 1/6`, `state = initialized`. Mỗi hit lidar: `score += 1/6` (tối đa
   1), track chưa confirmed thành `tentative`, và thành `confirmed` khi `score > 0.8` — tức
   sau 5 lần thấy liên tiếp (trên log: sinh ở frame 0, confirmed ở frame 4). Miss lidar
   trong FOV: `score −= 1/6`; track confirmed **giữ** trạng thái (1.0 → 0.833 vẫn
   confirmed). Xoá khi một trong các điều kiện đúng: `P[0,0]` hoặc `P[1,1] > max_P = 9`;
   track confirmed có `score < 0.6` (từ 1.0 cần 3 miss liên tiếp); track chưa confirmed
   có `score ≤ 0`. Chỉ lượt lidar đánh giá các điều kiện này.

## Bonus (không bắt buộc)

Liệt kê phần bonus đã làm, file bằng chứng trong `student/bonus/` và kết quả chính
(xem [RUBRIC.md](../RUBRIC.md) mục 2). Không làm thì ghi "Không".

- Không

## Khai báo sử dụng AI (bắt buộc)

Ghi rõ, kể cả khi không dùng ("Không dùng AI"). Xem [RULES.md](../RULES.md) mục 2.

- Công cụ đã dùng (ChatGPT, Copilot, Claude, …): Claude (Claude Code)
- Dùng cho phần nào (hàm, câu hỏi, debug): viết các hàm Part E–H trong `student/workspace/`
  (`kalman.py`, `camera_fusion.py`, `association.py`, `track_management.py`); viết script
  Kaggle để tải segment Waymo + weights và chạy lần chấm điểm (máy cá nhân không đủ dung
  lượng tải dữ liệu); soạn phần tóm tắt kết quả và 6 câu giải thích trong file này.
- Cách bạn đã kiểm tra lại (pytest, chạy Waymo, đối chiếu công thức): `pytest student/tests -q`
  trên Kaggle (128 passed, không `failed`/`xfailed`); chạy
  `fusion-run-lab --fusion compare --seed 0` trên cả segment (frame 0–198) trên Kaggle;
  `python tools/check_submission.py`; đối chiếu số liệu báo cáo với `metrics.json` và
  từng dòng `grade_run_lidar.log` / `grade_run_fused.log`.

## Checklist nộp

- [ ] **Part E–H** trong `workspace/` đã implement; `pytest student/tests -q` không còn `failed`/`xfailed`
- [ ] Part A–D: không bắt buộc sửa (hoặc ghi chú nếu bạn đã sửa)
- [ ] Lần chạy chấm điểm: `--fusion compare --seed 0`, `frame_start: 0`, `frame_end: 198`
- [ ] Đã commit `student/artifacts/metrics*.json` và `student/artifacts/grade_run*.log` (không sửa tay)
- [ ] Đã điền đủ file này, gồm khai báo AI
- [ ] Không commit dữ liệu Waymo, weights, `paths.yaml`, API key
- [ ] `python tools/check_submission.py` báo `KẾT QUẢ: SẴN SÀNG NỘP`
- [ ] Đã push và nộp link repo + commit hash trên LMS ([hướng dẫn nộp](../SUBMISSION.md))
