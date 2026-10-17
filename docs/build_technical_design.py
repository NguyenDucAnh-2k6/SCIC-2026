from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.units import mm

BASE = Path(__file__).resolve().parent
OUT = BASE / 'output' / 'pdf' / 'ClassTwin_Thiet_ke_du_lieu_va_ML_v1.pdf'
OUT.parent.mkdir(parents=True, exist_ok=True)
for name, file in [('Arial','arial.ttf'),('Arial-Bold','arialbd.ttf'),('Arial-Italic','ariali.ttf')]:
    pdfmetrics.registerFont(TTFont(name, str(Path('C:/Windows/Fonts') / file)))
pdfmetrics.registerFontFamily('Arial',normal='Arial',bold='Arial-Bold',italic='Arial-Italic',boldItalic='Arial-Bold')
NAVY=colors.HexColor('#102A43'); TEAL=colors.HexColor('#007F86'); GRAY=colors.HexColor('#526779'); LIGHT=colors.HexColor('#EDF5F7')
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyVN',fontName='Arial',fontSize=10,leading=14,spaceAfter=7,textColor=NAVY))
styles.add(ParagraphStyle(name='SmallVN',parent=styles['BodyVN'],fontSize=8.5,leading=11,spaceAfter=4))
styles.add(ParagraphStyle(name='H1VN',fontName='Arial-Bold',fontSize=20,leading=25,spaceAfter=12,textColor=NAVY))
styles.add(ParagraphStyle(name='H2VN',fontName='Arial-Bold',fontSize=11.5,leading=15,spaceBefore=9,spaceAfter=6,textColor=TEAL))
styles.add(ParagraphStyle(name='EqVN',fontName='Arial',fontSize=9.2,leading=14,spaceBefore=3,spaceAfter=8,leftIndent=10,textColor=NAVY,backColor=LIGHT,borderPadding=6))
styles.add(ParagraphStyle(name='Cover',fontName='Arial-Bold',fontSize=33,leading=40,spaceAfter=20,textColor=NAVY))
story=[]
def p(s, style='BodyVN'): story.append(Paragraph(s,styles[style]))
def h(s): p(s,'H2VN')
def eq(s): p(escape(s),'EqVN')
def bullets(items):
    for s in items: p('• '+s)
def table(headers,rows,widths):
    data=[[Paragraph(escape(str(s)).replace('\n','<br/>'),styles['SmallVN']) for s in row] for row in [headers]+rows]
    t=Table(data,colWidths=[x*mm for x in widths],repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),LIGHT),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),0.7,TEAL),('LINEBELOW',(0,1),(-1,-1),0.3,colors.HexColor('#D7E2E9')),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
    story.append(t); story.append(Spacer(1,7))
pages=[]
def page(title):
    if story: story.append(PageBreak())
    pages.append(title); p(title,'H1VN')

page('CLASS TWIN')
story.append(Spacer(1,14*mm))
p('Thiết kế dữ liệu<br/>và hệ thống học máy','Cover')
p('SCIC 2026 • Đặc tả kỹ thuật triển khai v1.0','H2VN')
p('Schema dữ liệu • Tiền xử lý • Formulation<br/>Training • Đánh giá • Inference • Triển khai Kaggle')
story.append(Spacer(1,12*mm))
p('<b>Bối cảnh:</b> chưa có cảm biến lớp học; dùng EdNet làm dữ liệu lịch sử cho mô hình người học. Source code phát triển bằng VS Code; tiền xử lý và huấn luyện chạy trên Kaggle.')
p('<b>Phạm vi:</b> thiết kế để hiện thực hóa project Python hoàn chỉnh. Các schema chuẩn hóa, ngưỡng, kiến trúc và cấu hình trong tài liệu là đề xuất kỹ thuật; chưa phải kết quả thực nghiệm.')
p('<b>Nguyên tắc bằng chứng:</b> phân biệt quan sát thật, trạng thái suy luận và phản ứng mô phỏng. Không suy diễn hiệu quả can thiệp giảng viên chỉ từ độ chính xác dự đoán EdNet.')
story.append(Spacer(1,12*mm))
p('Ngày lập: 17/10/2026 | Ngôn ngữ: Tiếng Việt<br/>Nguồn định hướng: đề án ClassTwin.pdf; tài liệu chính thức EdNet và các công trình [1]-[6].','SmallVN')

page('01 / Phạm vi và bản đồ tài liệu')
p('Mục tiêu sản phẩm là hỗ trợ giảng viên ra quyết định: quan sát → ước lượng trạng thái → mô phỏng → chọn gợi ý → giảng viên quyết định → ghi nhận phản hồi. AI không tự áp dụng can thiệp lên lớp.')
table(['Khối nội dung','Mục','Kết quả cần có'],[
('Dữ liệu','02-06','Hợp đồng dữ liệu, schema raw/canonical, ZIP → Parquet'),
('Đặc trưng và chia tập','07-08','Feature theo thời điểm, batch sequence không rò rỉ'),
('Formulation','09-12','State estimator, twin, POMDP và policy có ràng buộc'),
('Training và đánh giá','13-15','Thứ tự train, metric, ablation, điều kiện dùng RL'),
('Inference và vận hành','16-19','API, cache state, triển khai, tiêu chí nghiệm thu'),
('Nguồn và ký hiệu','20','Tài liệu tham khảo và quy ước diễn giải')],[43,22,109])
h('Ba mức triển khai')
bullets(['<b>M1 - EdNet:</b> ước lượng kiến thức, dự đoán câu trả lời, phát lại lịch sử. Đánh giá bằng dữ liệu giữ lại.', '<b>M2 - lớp mô phỏng:</b> ghép sinh viên thành lớp ảo, đặt giả định phản ứng với các can thiệp, so sánh policy bằng phân tích độ nhạy.', '<b>M3 - pilot thật:</b> tích hợp LMS, log giảng viên và cảm biến nếu có; hiệu chuẩn lại và thu dữ liệu để đánh giá policy thực tế.'])
h('Các quyết định khởi đầu')
p('Ưu tiên KT1 + Contents cho M1; KT3 là phần mở rộng. Dùng baseline đơn giản trước mô hình sâu. M1/M2 có thể hoàn thành trước khi mua phần cứng. Mục tiêu pilot theo đề án: một môn học, 1-3 lớp, 40-100 sinh viên/lớp trong một học kỳ [0].')

page('02 / Nguồn dữ liệu và giới hạn')
table(['Nguồn','Thông tin dùng được','Vai trò'],[
('EdNet KT1','Câu hỏi, đáp án của người học, thời gian và nhóm lần giải','Dự đoán kết quả; lịch sử học'),
('EdNet KT2/KT3','Sự kiện thao tác; KT3 thêm đọc giải thích và xem bài giảng','Dựng timeline hành vi'),
('Contents','Metadata câu hỏi/bài giảng, đáp án và tag kỹ năng','Ghép nhãn; mã hóa nội dung'),
('LMS/Teacher App tương lai','Quiz lớp, hoạt động giảng viên thực thi và phản hồi','Hiệu chuẩn trên lớp thật'),
('Cảm biến tương lai','CO2, nhiệt độ, độ ồn tổng hợp','Context môi trường, có mask thiếu')],[36,85,53])
p('EdNet là lịch sử tương tác cá nhân trong hệ thống học Santa; không phải bản ghi lớp học đồng bộ. Dữ liệu tổ chức theo người dùng và timestamp đã được dịch chuyển. Không dùng mốc lịch để suy luận giờ học thật hoặc đồng bộ sinh viên thành lớp [1,2].')
h('Không đồng nhất các khái niệm')
bullets(['Đáp án đúng là quan sát; mastery là suy luận từ lịch sử và độ khó câu hỏi.', 'Thời gian không hoạt động, đổi đáp án hoặc thoát bài chỉ là proxy hành vi. Không có nhãn xác nhận attention, fatigue hay động lực.', 'Không gán chuỗi thao tác người học thành chuỗi quyết định của giảng viên. Xem video không đồng nghĩa giảng viên đang thuyết trình.', 'Thiếu cảm biến được biểu diễn bằng null + mask, không điền số đo giả vào trường quan sát.'])
h('Chính sách bằng chứng trong sản phẩm')
p('Mọi bản ghi mang provenance: observed, derived, inferred hoặc simulated. Dashboard phải hiển thị chế độ replay/simulation/pilot; kết quả mô phỏng ghi rõ giả định và phiên bản simulator. Không báo “30% mất tập trung” khi chỉ có proxy hành vi.')

page('03 / Schema raw EdNet')
p('Đây là các trường raw cần cho adapter. Kiểm tra header của archive thật trước khi ánh xạ; không suy kiểu cột theo vị trí. README có bảng ví dụ KT1 không nhất quán với mô tả, nên dùng tên cột thực tế và kiểm tra khóa [1].')
table(['Bảng','Trường / kiểu raw','Ý nghĩa và xử lý'],[
('KT1 CSV','timestamp: int64\nsolving_id: int\nquestion_id: string\nuser_answer: string\nelapsed_time: int64','Thời điểm, nhóm giải, câu hỏi, đáp án và thời lượng. Đơn vị thời gian ms. user_id lấy từ tên file.'),
('KT3 CSV','timestamp: int64\naction_type: string\nitem_id: string\nsource: string\nuser_answer: string/null\nplatform: string','Sự kiện theo người dùng. item_id có thể chỉ câu hỏi, bundle, giải thích hoặc bài giảng. Không join mọi item_id vào question_id.'),
('questions.csv','question_id, bundle_id, explanation_id: string\ncorrect_answer: string\npart: int\ntags: string\ndeployed_at: int64','Khóa question_id; tags tách theo dấu chấm phẩy. Tạo is_correct bằng so sánh đáp án.'),
('lectures.csv','lecture_id: string\npart: int\ntags: int\nvideo_length: int64\ndeployed_at: int64','Dùng khi mở rộng KT3. Một số metadata có -1; chuẩn hóa thành null + cờ thiếu.')],[30,71,73])
h('Quy tắc đọc')
p('Giữ ID dạng chuỗi để bảo toàn tiền tố. Không giả định mọi câu đều có bốn lựa chọn. Các câu trong bundle liên quan cùng ngữ liệu; giữ nguyên nhóm khi chia sequence và dự đoán trước khi thấy kết quả của bundle đó [1].')
p('Tài liệu này không giả định archive đã có mặt trên máy hoặc đã được kiểm tra thống kê. Báo cáo profiling và manifest là đầu ra bắt buộc của lần chạy đầu tiên.')

page('04 / Schema chuẩn hóa cho M1')
p('Lưu Parquet theo shard, không theo từng sinh viên. Trường có dấu * bắt buộc; null và mask được giữ đồng nhất giữa training và inference. Mỗi file có schema_version trong metadata.')
table(['Bảng','Khóa và trường chính','Bất biến'],[
('content_items','item_id* string; item_type* enum; bundle_id string?; skill_ids list<int>; part int?; correct_answer string?; available_at_ms int64?','Khóa duy nhất theo item_type + item_id. Danh mục cố định không đồng nghĩa nội dung đã có ở mọi thời điểm.'),
('events','event_id* string; student_id* string; event_ts_ms* int64; source_row* int64; event_type* enum; item_id string?; response string?; source string?; platform string?; provenance* enum','Một dòng là một sự kiện. Duy trì thứ tự nguồn cho timestamp bằng nhau; thời gian nhận sự kiện thật lưu riêng.'),
('attempts','attempt_id* string; student_id*; question_id*; bundle_instance_id*; attempt_index* int32; observed_ts_ms* int64; answered_at_ms int64?; is_correct bool?; elapsed_ms int64?; timing_semantics* enum; quality_flags list<string>','Nhãn chỉ có khi trả lời hợp lệ và metadata khớp. Không tự đổi timestamp KT1 thành thời điểm submit.'),
('student_sessions','session_id*; student_id*; start_ms*; end_ms*; boundary_reason*; split* enum','Session là quy tắc kỹ thuật; không phải buổi học thật. Giữ nhóm bundle cùng session.')],[30,98,46])
h('Nhận diện bundle và tính đúng/sai')
p('KT1: bundle_instance_id kết hợp student_id và solving_id, kiểm tra tương ứng với metadata bundle_id. KT3: tách từng lần enter/submit cùng bundle. Đáp án cuối là respond cuối trước submit của câu hỏi; thiếu submit không tự coi là hoàn tất. Giữ các lần làm lại khác thời điểm thành attempt mới.')

page('05 / Schema twin và nhật ký quyết định')
table(['Bảng','Trường cốt lõi đề xuất','Cách dùng'],[
('student_state','student_id; as_of; model_version; skill_probs; support_counts; behavior_features; uncertainty; last_event_id; provenance','Snapshot sau lịch sử quan sát. skill_probs là dự đoán có điều kiện, không phải nhãn mastery thật.'),
('class_snapshot','class_id; decision_id; as_of; cohort_version; topic_id; n_enrolled; n_observed; n_eff; aggregate_features; missing_masks; mode','Lớp mô phỏng có cohort_version và seed. n_eff là số người có dữ liệu còn mới.'),
('env_context','class_id; event_ts; received_at; co2_ppm?; temperature_c?; noise_db?; missing_masks; quality_flags','Hiện để thiếu. Không suy ra cảm biến từ thời gian làm bài.'),
('teacher_decisions','decision_id; class_id; state_ref; suggested_action; executed_action?; planned_duration_s; actual_duration_s?; accepted_status; override_reason?; policy_version; support_score; behavior_propensity?','Tách gợi ý và hành động thực thi. Chỉ lưu propensity nếu có xác suất hợp lệ của cơ chế chọn hành động.'),
('transitions','episode_id; step; state_ref; executed_action; reward?; reward_components; next_state_ref; duration_s; terminated; truncated; data_origin; simulator_version?','Chỉ tạo khi đủ ngữ nghĩa (s,a,r,s′). data_origin = observed hoặc simulated; không trộn âm thầm.')],[32,99,43])
h('Hợp đồng quan trọng')
p('Mỗi decision_id duy nhất; thực thi có thể đến sau đề xuất. Reject không phải can thiệp lớp; transition phải gắn hành động thật hoặc trạng thái chưa biết. Log phản hồi giảng viên hữu ích cho thiết kế sản phẩm, nhưng accept không phải nhãn “hành động tốt”.')
p('Với dữ liệu thật, ID sinh viên dùng bí danh; khóa ánh xạ định danh nằm ngoài artifact training. Đề án không thu camera thô hoặc âm thanh định danh [0].')

page('06 / Tiền xử lý ZIP → Parquet')
h('Bước 1. Kiểm kê và tạo manifest')
p('Ghi archive SHA-256, tên/phiên bản nguồn, kích thước, danh sách member hợp lệ, cấu hình và commit code. Sắp tên file xác định; kiểm tra CRC khi đọc. Không bung toàn bộ archive vào workspace IDE. Bộ nhớ danh mục ZIP vẫn tăng theo số file, nên profiling cả bước mở archive.')
h('Bước 2. Đọc theo dòng, ghi theo lô')
p('Mở member bằng ZipFile.open; đọc CSV theo chunk và schema định trước. Lấy student_id từ đường dẫn đã chuẩn hóa. Mục tiêu khởi đầu: 100.000-500.000 dòng/lô và shard Parquet khoảng 128-256 MB, điều chỉnh theo RAM thực đo. Không giữ toàn dataset trong một DataFrame; không mở một writer cho mỗi sinh viên.')
h('Bước 3. Chuẩn hóa và kiểm định')
bullets(['Ép kiểu; chuyển sentinel thành null; kiểm tra đáp án theo metadata. Join thiếu đưa vào bảng quarantine kèm lý do; không im lặng gán incorrect.', 'Giữ raw row index. Chỉ loại bản ghi trùng có cùng nguồn/event identity; không drop mọi dòng có cùng timestamp hoặc cùng câu hỏi.', 'Đọc chuỗi theo người dùng, kiểm tra timestamp không giảm. Nếu cần external sort, ghi shard tạm rồi sort có giới hạn RAM.', 'Thời lượng âm hoặc lỗi parse bị gắn cờ. Thời lượng 0/ngoại lệ không tự xóa; thống kê, tạo mask và dùng robust transform.'])
h('Bước 4. Xuất dữ liệu có thể tiếp tục chạy')
p('Ghi shard tạm, đóng writer, kiểm tra row count rồi rename để commit. Ledger lưu member đã xử lý, checksum và output shards. Khi resume, bỏ qua member đã commit; tránh ghi lặp. Shard chỉ commit theo nhóm member hoàn chỉnh để đơn giản hóa khôi phục.')
eq('raw ZIP → adapters → canonical attempts/events → split manifest → features/sequences')
p('Đầu ra: manifest.json, quality_report.json, quarantine/*.parquet, attempts/*.parquet, content/*.parquet. Partition theo split và bucket hash(student_id) ổn định; không tạo hàng trăm nghìn partition.')

page('07 / Thời gian, session và đặc trưng')
h('Nguyên tắc as-of')
p('Feature tại quyết định t chỉ dùng dữ liệu đã quan sát tới t. Với KT1, thứ tự trong bundle không đủ chứng minh thứ tự biết kết quả; mặc định dự đoán toàn bộ bundle trước khi cập nhật đáp án của bundle. Thời gian làm câu hiện tại không được dùng để dự đoán đáp án trước khi sinh viên làm xong.')
table(['Nhóm','Đặc trưng đề xuất','Xử lý thiếu / giới hạn'],[
('Kết quả quá khứ','Số lần thử; tỉ lệ đúng có làm trơn; chuỗi sai; kết quả gần nhất theo skill','Chỉ history; unseen skill dùng prior train và support=0'),
('Thời gian','log(1+elapsed_ms); thời gian từ tương tác trước; độ dài session đã qua','Fit clipping/scale trên train. Không cộng elapsed của mọi câu trong bundle nếu có dấu hiệu thời lượng dùng chung'),
('Hành vi KT3','Số lần đổi đáp án; số lượt xem giải thích; thời lượng enter-quit hợp lệ','Chỉ tính sau sự kiện hoàn tất; enter/quit thiếu tạo cờ, không tự điền 0'),
('Nội dung','Embedding item/skill/part; độ khó thống kê trên train','Unknown ID có mã UNK; tách content identity khỏi tương tác kết quả'),
('Chất lượng','Missing mask; support_count; độ cũ của state; số sự kiện lỗi','Là đầu vào và điều kiện từ chối gợi ý')],[29,88,57])
h('Session đề xuất')
p('Tách khi khoảng cách giữa các nhóm sự kiện vượt 30 phút; đây là siêu tham số, kiểm tra thêm 15/60 phút. Ưu tiên ranh giới bundle; không chẻ đôi nhóm để đạt độ dài mong muốn. Khoảng trống lớn không được mặc định là fatigue hoặc mất tập trung.')
h('Ghép sự kiện xem nội dung')
p('KT3: ghép enter-quit theo người dùng và loại nội dung bằng state machine. Enter lặp, quit thiếu hoặc thứ tự bất thường có quality_flag. Không đếm trùng thời gian của các interval chồng nhau. Lưu timing_semantics để phân biệt thời lượng đọc/video với elapsed của KT1.')

page('08 / Chia tập và tạo sequence')
h('Protocol A - khái quát sang sinh viên mới')
p('Chia student_id bằng hash có seed, ví dụ 80/10/10 train/validation/test. Toàn bộ dữ liệu của một người chỉ thuộc một split. Danh sách người học và seed được lưu cố định. Fit vocab, độ khó item, scale, clipping và prior chỉ trên train; validation dùng chọn model và calibration; test mở sau khi chốt.')
h('Protocol B - dự báo tương lai cùng người học')
p('Một thí nghiệm riêng: chia lịch sử mỗi người theo thời gian, giữ bundle nguyên vẹn. Khi đánh giá suffix, có thể warm-start bằng prefix quan sát, nhưng không dùng nhãn tương lai. Không trộn kết quả protocol A và B thành một chỉ số. Timestamp đã dịch chuyển không thích hợp cho đánh giá lịch học thực tế giữa người dùng.')
h('Tạo batch cho sequence model')
eq('Input trước bundle b: H_(b-1), metadata(Q_b) → p(y_q = 1 | H_(b-1), q)')
eq('Sau khi có phản hồi bundle b: H_b = update(H_(b-1), responses_b)')
p('Lưu sequence theo student/session; embedding multi-skill dùng pooling có mask. Điểm bắt đầu: history tối đa 200 câu, batch 64; điều chỉnh theo VRAM. Không chẻ bundle. Có thể dùng context chồng lấn giữa window, nhưng label mask chỉ tính mỗi target một lần. PAD và UNK dùng mã khác nhau.')
bullets(['Target câu chưa có nhãn hoặc đáp án lỗi: loss_mask=0; không gán y=0.', 'Teacher forcing: quan sát lịch sử thật cho dự đoán một bước. Rollout nhiều bước phải tự lấy mẫu phản hồi và báo sai số tích lũy riêng.', 'Không đưa user_id embedding riêng vào baseline cần tổng quát hóa sang người mới; trạng thái khởi tạo từ prior.', 'Các bản KT1/KT3 có tương tác chồng lặp: chọn một nguồn nhãn chính; không nối hai nguồn làm tăng số lần trả lời.'])

page('09 / Formulation: state estimator')
p('Ký hiệu: i là người học; b là bundle; q là câu hỏi; k là kỹ năng. H là lịch sử đã biết. Mô hình dự đoán kết quả quan sát; không coi latent embedding là đo lường trực tiếp trạng thái tâm lý.')
eq('p_(i,q) = P(y_(i,q)=1 | H_i, item_q, skills_q)')
eq('L_KT = - sum_mask [ y log(p) + (1-y) log(1-p) ] / N_valid')
h('Mô hình và baseline')
p('Bắt đầu bằng prior theo item/skill và logistic regression với feature lịch sử. Sau đó triển khai BKT theo kỹ năng như baseline diễn giải và GRU/DKT có xử lý multi-skill. DKT là họ mô hình truy vết kiến thức từ chuỗi tương tác [3]; thiết kế bundle trong tài liệu này là lựa chọn triển khai riêng.')
h('Định nghĩa mastery cho dashboard')
p('Nếu dùng BKT: hiển thị posterior mô hình theo skill với giới hạn giả định. Nếu dùng GRU: dùng xác suất đúng trung bình trên tập câu probe cố định theo skill, có kiểm soát độ khó. Tên hiển thị nên là “mức nắm kiến thức ước lượng”; không so sánh trung bình xác suất trên các tập câu có độ khó khác nhau.')
eq('m_(i,k) = mean_{q in Probe(k)} p(y=1 | H_i, q)')
p('Probe chọn từ nội dung có sẵn, không dùng đáp án thực của test để cập nhật hoặc chọn probe. Freeze probe_version. Với kỹ năng ít quan sát, hiển thị support_count và uncertainty cao, không gán kết luận chắc chắn.')
h('Engagement và fatigue')
p('M1 chỉ xuất behavior_features (nhịp tương tác, độ dài session, khoảng gián đoạn). Không train head fatigue/engagement bằng nhãn tự tạo rồi báo như ground truth. Nếu M2 cần các biến này, dùng latent giả định trong simulator; M3 cần nhãn khảo sát/quan sát hợp lệ để hiệu chuẩn.')

page('10 / Formulation: Classroom Digital Twin')
h('Động lực học người học')
eq('z_(i,t+1) ~ T_theta(z_(i,t), content_t, delta_t, context_t)')
eq('o_(i,t+1) ~ O_theta(z_(i,t+1), content_(t+1))')
p('Trong M1, z là trạng thái ẩn của estimator. T được học thông qua dự đoán chuỗi trên dữ liệu EdNet. Có thể thêm head dự đoán log-thời lượng hợp lệ bằng Huber loss với mask. Chỉ bật loss phụ sau khi xác minh elapsed ở mức câu/bundle và tránh gán thời lượng bundle thành nhiều nhãn độc lập.')
h('Tách mô hình quan sát và mô hình can thiệp')
eq('M2: z_(i,t+1) ~ T_(theta,phi)(z_(i,t), a_t, content_t, delta_t, context_t)')
p('theta học từ dữ liệu; phi mô tả tác động can thiệp chưa quan sát. EdNet không xác định được phi cho thảo luận nhóm hay giải lao. Đặt phi thành tập kịch bản: không tác động, tác động nhỏ/lớn, không đồng đều theo nhóm và có thể bất lợi. Ghi giả định, nguồn chuyên gia nếu có, seed và version.')
h('Tổng hợp thành lớp')
p('Tạo cohort mô phỏng 40-100 người từ cùng split; căn theo thời gian tương đối kể từ đầu episode. Mỗi lớp giữ phân phối năng lực, không chỉ lấy trung bình. Features gồm mean/quantile mastery theo chủ đề, tỉ lệ dưới ngưỡng, coverage, độ phân tán và độ cũ state. Lớp ảo không được mô tả là lớp thật.')
h('Mô phỏng phản thực')
p('Từ cùng snapshot, chạy nhiều rollout cho từng hành động với cùng tập seed nhiễu để so sánh. Báo median, khoảng phân vị mô phỏng và độ nhạy theo phi. Khoảng này là độ biến thiên theo simulator, không phải bảo đảm hiệu quả trên lớp thật. Giới hạn horizon để tránh tích lũy lỗi; ablation ensemble và horizon là bắt buộc.')

page('11 / Formulation: bài toán quyết định')
p('Mô hình lớp học là POMDP hữu hạn thời gian: trạng thái thật s không quan sát trực tiếp; belief b_t = P(s_t | lịch sử quan sát và hành động). Thực tế dùng vector ước lượng x_t = f(H_≤t), chỉ là xấp xỉ belief và có thể chưa Markov.')
table(['Thành phần','Định nghĩa triển khai'],[
('Observation o_t','Kết quả quiz trước đó, lịch sử tương tác, phản hồi giáo viên, context và missing masks.'),
('State x_t','Phân phối mastery theo kỹ năng/nhóm, proxy hành vi, uncertainty, thời gian còn lại, hành động trước và coverage.'),
('Action a_t','Loại hoạt động + chủ đề + thời lượng dự kiến; mask theo thời gian, nội dung sẵn và giới hạn workload.'),
('Episode','Một buổi học mô phỏng, thí dụ 60 phút; M3 dùng ranh giới buổi thật. Episode reset state theo giao thức công bố.'),
('Transition','Chỉ gắn hành động thực thi. Delta_t là thời gian hành động thực tế; thiếu action/outcome thì transition không đủ điều kiện RL.')],[33,141])
h('Không gian hành động M2 đề xuất')
p('Năm loại: tiếp tục giảng, quiz ngắn, bài tập nhóm, thảo luận, giải lao. MVP cố định từng thời lượng trong config, ví dụ 5/3/5/5/3 phút; đây là lựa chọn mô phỏng chưa hiệu chuẩn. Các hành động này không phải action labels có sẵn trong EdNet.')
h('Thời lượng thay đổi')
eq('gamma_t = gamma_base ** (delta_t / tau), với tau = 60 giây')
p('Dùng semi-Markov formulation hoặc quyết định theo bước thời gian cố định. Nếu dùng discount theo thời lượng, áp dụng nhất quán trong rollout, TD target và đánh giá. terminated là hết buổi; truncated là dừng kỹ thuật và cần xác định có bootstrap theo giao thức hay không.')

page('12 / Reward, constraints và Offline RL')
eq('r_t = w_m * Delta_M_t + w_l * Delta_M_low,t - w_c * workload_t - w_s * switch_t')
p('Delta_M là thay đổi mastery trên cùng bộ probe, Delta_M_low là mức thay đổi của nhóm yếu được xác định tại đầu episode. Đây là reward mô hình trong M2; phải đánh giá thêm kết quả quiz giữ lại khi có dữ liệu thật. Không dùng reward này làm bằng chứng độc lập về kiến thức thật tăng lên.')
eq('maximize_pi E[sum_t gamma_(0:t) r_t]')
eq('subject to E[sum workload_t] <= B; E[sum harm_low,t] <= epsilon')
p('Đề xuất harm_low,t = max(0, -Delta_M_low,t). Thêm metric chênh lệch lợi ích giữa nhóm, số lần đổi hoạt động và tỉ lệ vi phạm giới hạn thời gian. Đây là fairness theo nhóm năng lực, không phải chứng minh công bằng toàn diện. Chuẩn hóa reward và cost trước khi chọn trọng số; config lưu rõ ngân sách và đơn vị.')
h('M1 chưa tạo được dataset Offline RL cấp lớp')
p('Không chuyển trực tiếp KT1 thành (state, hành động giảng viên, reward, next state). M2 có thể tạo dataset tổng hợp bằng nhiều behavior policy, gồm quy tắc và ngẫu nhiên có ràng buộc; đánh giá CQL trên dữ liệu tổng hợp phải ghi rõ “offline RL trong simulator”.')
h('Mục tiêu CQL rời rạc tham khảo [4]')
eq('y_TD = r + gamma_t * (1-d) * max_a Q_target(x_next, a)')
eq('L = 0.5 E_D[(Q(x,a)-y_TD)^2]')
eq('    + alpha * E_x[log sum_{a in A_valid(x)} exp Q(x,a) - Q(x,a_D)]')
p('Dùng action mask cả TD target lẫn policy; d theo quy ước termination. State phải thêm workload đã dùng và ngân sách còn lại; mask action vượt trần chi phí dự kiến. Ràng buộc harm được sàng lọc bằng mô phỏng và kiểm tra sau train, chưa được bảo đảm trên lớp thật. CQL không sửa được thiếu causal identification hoặc action support [4]. M3 chỉ train lại khi log can thiệp/outcome đủ chất lượng.')

page('13 / Pipeline training từ đầu đến cuối')
table(['Stage','Đầu vào → công việc','Artifact / điều kiện chuyển'],[
('P0 • Audit','ZIP + Contents → schema, CRC, row count, null, bundle/timing profiling','manifest; quality report; quarantine có lý do'),
('P1 • Dataset','Canonical tables → split cố định, transform fit train, sequence/masks','split manifest; vocab; scaler; feature spec'),
('P2 • Estimator','Train baseline → BKT/GRU; early stopping trên validation log loss','Checkpoint tốt nhất; metric theo nhóm; model card'),
('P3 • Calibration','Validation holdout riêng → temperature scaling hoặc phương pháp phù hợp','calibration artifact; Brier/ECE; probe_version'),
('P4 • Twin','Estimator đóng băng + kịch bản phi → ensemble và cohort ảo','simulator_version; scenario grid; rollout report'),
('P5 • Policy','Synthetic transitions → rule, BC, CQL; tune validation scenarios','policy checkpoint; reward/cost config; support report'),
('P6 • Release','Test khóa kín; kiểm tra artifact bundle và replay inference','release manifest; thresholds; fallback; báo cáo giới hạn')],[29,82,63])
h('Cấu hình khởi đầu, chưa phải tối ưu')
p('GRU: embedding 64, hidden 128, dropout 0,1; AdamW, learning rate 0,001; batch 64; max history 200; tối đa 30 epoch, patience 5 theo validation log loss. Gradient clipping 1,0; seed 3 giá trị. Giảm batch nếu thiếu VRAM, không thay ngẫu nhiên protocol split.')
p('CQL M2: MLP 2 lớp 128; batch 256; learning rate 0,0003; target network cập nhật theo config; thử alpha ∈ {0,1; 1; 5}. Chọn bằng lợi ích và vi phạm constraints trên validation simulator khác seed. Tham số discount, horizon và action duration phải được chốt cùng episode config.')
p('Không huấn luyện đồng thời estimator và policy ở bản đầu: thay estimator sẽ làm đổi state/reward. Mỗi policy phải pin đúng estimator, probe, transforms và simulator đã dùng khi train.')

page('14 / Đánh giá estimator và twin')
table(['Hạng mục','Metric / giao thức','Cách diễn giải'],[
('Dự đoán đúng/sai','Log loss chính; ROC-AUC, PR-AUC khi cần, Brier; macro theo người + micro theo sự kiện','Accuracy chỉ phụ; AUC không xác nhận mastery thật.'),
('Hiệu chuẩn','Reliability diagram, ECE theo bin công bố; Brier trước/sau calibration','Giữ calibration subset tách khỏi tập chọn checkpoint khi đủ dữ liệu.'),
('Theo nhóm','Ít/nhiều history; kỹ năng; part; độ dài session; nhóm năng lực đầu kỳ','Luôn kèm số người, số label và missing rate; không chọn nhóm theo điểm test tương lai.'),
('Thời lượng','MAE trên log-time, median absolute error theo timing_semantics','Chỉ trên nhãn hợp lệ; báo tỷ lệ nhãn bị mask.'),
('Twin quan sát','NLL một bước; rollout 1/5/10 bundle; sai số phân phối đúng và thời lượng','Độ khớp hành vi quan sát không xác nhận phản ứng can thiệp.'),
('Bất định','Coverage của interval khi có target quan sát; ensemble disagreement; OOD checks','Phân biệt biến thiên lấy mẫu, bất định mô hình và bất định giả định.')],[31,84,59])
h('Ablation tối thiểu')
p('So sánh history-only với thêm time; KT1 với KT3 trên tập người/target giao nhau; có/không đặc trưng chất lượng; BKT với GRU; ensemble với single model; độ dài history và protocol bundle. Giữ cùng split, target và budget để so sánh công bằng.')
h('Khoảng tin cậy và nguồn phụ thuộc')
p('Bootstrap theo sinh viên hoặc cohort, không bootstrap từng dòng như độc lập. Báo mean và độ phân tán qua seed; test không dùng chọn kiến trúc, ngưỡng, phi hay reward. Với dữ liệu EdNet độc lập, không kết luận về hiệu quả trên môn học/lớp Việt Nam trước hiệu chuẩn pilot.')

page('15 / Đánh giá policy và điều kiện dùng')
h('M2 - đánh giá trong mô phỏng')
p('Baseline gồm: kế hoạch hoạt động cố định, quy tắc theo coverage/mastery, ngẫu nhiên có ràng buộc và behavior cloning trên dữ liệu tổng hợp. Train policy và đánh giá bằng seed/cohort tách biệt; giữ thêm scenario phi chưa dùng tune. Không chỉ chọn policy theo reward trên đúng simulator sinh dữ liệu train.')
bullets(['Báo return, gain nhóm yếu, workload, tỷ lệ vi phạm constraints, số lần đổi hoạt động, tỷ lệ fallback và mức hỗ trợ hành động.', 'Stress test: tác động can thiệp bằng 0, giả định sai dấu, missing observations, lớp phân hóa mạnh, độ trễ cập nhật và drift độ khó.', 'Chỉ dùng kết quả để so sánh dưới giả định simulator. Đánh giá bằng simulator thứ hai hoặc tham số nhiễu giúp kiểm tra độ bền, không thay thế pilot thật.'])
h('M3 - off-policy evaluation từ log thật')
p('Trước OPE cần action space tương thích, state trước can thiệp, executed_action, outcome đúng cửa sổ, đủ overlap và giả định không có nhiễu ẩn quan trọng. Dữ liệu giảng viên quan sát được nhưng không ghi vào state có thể gây confounding. Cần nêu giả định này trong báo cáo.')
p('Fitted Q Evaluation (FQE) là một lựa chọn [5], nhưng không đáng tin cho hành động chưa có hỗ trợ. IS/DR chỉ dùng khi behavior propensity đã biết hoặc được ước lượng đáng tin và kiểm tra trọng số/ESS; không lấy điểm tự tin của policy làm propensity của giảng viên. Bootstrap theo lớp/buổi thay vì theo transition.')
h('Điều kiện phát hành đề xuất')
p('Estimator phải cải thiện hoặc ít nhất không suy giảm baseline theo protocol đã chốt, có calibration chấp nhận được. Policy phải đạt budget workload, không gây giảm gain nhóm yếu vượt epsilon và vượt stress tests đã định nghĩa. Nếu chưa đủ log thật, chỉ phát hành chế độ mô phỏng hoặc shadow; không gắn nhãn “đã tối ưu cho lớp thật”.')

page('16 / Inference trực tuyến đầy đủ')
table(['Bước','Xử lý tại runtime'],[
('1. Nhận sự kiện','Validate schema/version; gắn received_at; kiểm tra event_id để chống trùng. Event lỗi vào quarantine, không cập nhật state.'),
('2. Cập nhật history','Sắp theo event time trong cửa sổ trễ cấu hình. Với bundle, đợi feedback hoàn tất; hết timeout thì giữ trạng thái thiếu nhãn.'),
('3. Cập nhật người học','Nạp cache đúng model_version; gọi cùng feature transforms và estimator như train. Ghi last_event_id và as_of. Sự kiện đến quá trễ kích hoạt replay từ checkpoint.'),
('4. Tổng hợp lớp','Theo roster thật hoặc cohort mô phỏng. Tính coverage/n_eff, freshness và uncertainty; không bỏ sinh viên thiếu dữ liệu ra khỏi mẫu số mà không báo.'),
('5. Sinh ứng viên','Mask hành động theo thời gian còn lại, nội dung sẵn, cooldown, workload và support. Thiếu bằng chứng hoặc mode sai thì fallback.'),
('6. Chọn và giải thích','Rule hoặc policy đúng chế độ; tùy chọn rollout twin. Trả gợi ý, phạm vi bằng chứng, các kịch bản và lý do dựa trên feature thật.'),
('7. Giảng viên quyết định','Ghi suggested_action, accept/modify/reject; khi thực thi mới ghi executed_action và actual_duration.'),
('8. Đóng vòng phản hồi','Sau cửa sổ outcome, lưu observation và transition hợp lệ. Dùng cho batch retraining có version; không tự cập nhật trọng số online.')],[30,144])
h('Cold start và dữ liệu thiếu')
p('Sinh viên mới dùng prior theo nội dung + support=0; không xuất mastery quá chắc chắn. Không có cảm biến: truyền missing mask. Khi coverage thấp hoặc state cũ, trả “chưa đủ dữ liệu” hoặc quy tắc do giảng viên chọn. Không cập nhật state bằng response do twin tự sinh trong mode pilot.')
h('Tính nhất quán trạng thái')
p('Cập nhật theo thứ tự tuần tự mỗi sinh viên; snapshot và last_event_id commit cùng nhau. Model mới có thể không tương thích hidden state cũ: replay lịch sử giữ lại hoặc reset có cờ, không tái dùng cache tùy tiện. Thử nghiệm offline-online parity trên cùng chuỗi sự kiện trước phát hành.')

page('17 / API, phản hồi và fallback')
h('Các endpoint đề xuất')
table(['Endpoint','Yêu cầu → phản hồi'],[
('POST /v1/events:batch','events, schema_version → accepted_ids, duplicate_ids, rejected với lý do'),
('GET /v1/classes/{id}/state','as_of → mastery_summary, coverage, n_eff, behavior_proxies, missing_masks, uncertainty, versions'),
('POST /v1/recommendations','class_id, topic, decision_id, mode, time_remaining → gợi ý hoặc abstain, alternatives, support, explanations'),
('POST /v1/teacher-decisions','decision_id, status, executed_action?, actual_duration? → log xác nhận và trạng thái chờ outcome'),
('POST /v1/simulations','snapshot_ref, candidate_actions, scenario_set → phân phối outcome mô phỏng và assumption_version')],[66,108])
h('Ví dụ response rút gọn, không phải số liệu thực nghiệm')
eq('{ mode: "simulation", decision_id: "demo-001",')
eq('  recommended_action: "quiz", duration_s: 180,')
eq('  coverage: 0.82, evidence_level: "simulated",')
eq('  expected_gain: {median: 0.02, p10: -0.01, p90: 0.05},')
eq('  requires_teacher_decision: true, versions: {...} }')
p('Response thực phải có đơn vị gain, probe_version và scenario_set; p10/p90 là phân vị mô phỏng, không gọi là xác suất cải thiện thật. Giải thích nêu nguồn, chẳng hạn “nhóm kỹ năng X có nhiều dự đoán dưới ngưỡng, nhưng còn 18% thiếu dữ liệu”; tránh khẳng định học viên đang mệt.')
h('Fallback và ngân sách thời gian')
p('Cấu hình ngưỡng coverage/freshness/uncertainty trên validation và thử tải. Mục tiêu kỹ thuật ban đầu: state API p95 dưới 1 giây, recommendation dưới 3 giây với snapshot cache; đây là mục tiêu đo sau triển khai. Twin vượt timeout thì trả rule hoặc abstain, kèm reason_code. Không chặn tiến trình lớp để đợi mô phỏng.')

page('18 / Cấu trúc project và chạy Kaggle')
table(['Module / thư mục','Trách nhiệm'],[
('src/classtwin/data/','adapters, schemas, zip_reader, validation, preprocessing, features, sequences'),
('src/classtwin/state_estimation/','baselines, bkt/gru, calibration, state aggregation'),
('src/classtwin/twin/','learner dynamics, classroom environment, scenarios, rollout'),
('src/classtwin/policies/','rules, behavior cloning, cql, action masks, constraints'),
('src/classtwin/training/ và evaluation/','train loops, checkpoint, metrics, OPE và reporting'),
('src/classtwin/inference/ và utils/','API, state store, transforms; config, seed, logging, hashing'),
('configs/, scripts/, notebooks/, tests/','Cấu hình; CLI; notebook runner mỏng; kiểm thử hợp đồng dữ liệu')],[80,94])
h('Luồng VS Code → Kaggle')
p('Code và config lưu Git; đóng gói source.zip cùng commit manifest, không chứa raw data, .venv, token hoặc checkpoint. Kaggle gắn dataset source và dataset EdNet riêng; giải nén source nhỏ vào /kaggle/working/project, cài package từ pyproject.toml, rồi gọi CLI. Không bung EdNet thành hàng trăm nghìn CSV. /kaggle/input chỉ dùng đọc; /kaggle/working chứa đầu ra cần lưu phiên bản [6].')
eq('python scripts/preprocess.py --config configs/preprocess.yaml')
eq('python scripts/train_estimator.py --config configs/estimator.yaml')
eq('python scripts/train_policy.py --config configs/policy_sim.yaml')
p('Các lệnh là giao diện cần triển khai, chưa tồn tại trong repository. Config hỗ trợ raw_zip, content_dir, output_dir và artifact_dir; không hardcode mount dataset. Sau preprocessing, lưu Parquet thành dataset phiên bản riêng. Training gắn Parquet, không làm lại preprocessing mỗi lần.')
p('Khóa phiên bản dependency, ghi pip freeze và GPU/runtime thực tế. Internet tắt thì dùng wheel tương thích được gắn riêng. Checkpoint theo epoch kèm optimizer, scheduler, RNG state; chỉ có checkpoint model là chưa đủ resume chính xác. Kiểm tra output đã được lưu qua phiên bản Kaggle trước khi kết thúc phiên.')
p('Kaggle dùng cho batch training, replay và demo. API pilot cần triển khai riêng trên máy chủ trường hoặc cloud với state store bền vững; không phụ thuộc phiên notebook để phục vụ lớp đang học. Chuyển release bundle đã kiểm tra sang inference server, không chuyển toàn bộ EdNet.')

page('19 / Kiểm thử và tiêu chí nghiệm thu')
table(['Nhóm kiểm tra','Điều kiện cần đạt'],[
('Tính toàn vẹn dữ liệu','Số dòng đọc = hợp lệ + quarantine + duplicate có lý do; FK join báo coverage; ID ổn định; cùng đầu vào/config sinh cùng manifest logic.'),
('Không rò rỉ','Không giao student giữa split A; không target tương lai trong feature; không chia bundle; không target bị đếm nhiều lần qua window overlap.'),
('ZIP và resume','Archive mẫu có CSV lỗi/thiếu metadata/CRC lỗi được báo rõ; dừng rồi resume không lặp hoặc mất member đã commit.'),
('Estimator','Loss chỉ tính label hợp lệ; unseen item/skill không crash; calibration và baseline report có sample count.'),
('Twin và policy','Action mask hợp lệ; hết thời gian không phát sinh action dài quá buổi; reward/cost unit test bằng tình huống tính tay; simulator seed tái lập.'),
('Inference','Batch/online cùng prefix cho cùng output trong tolerance; event trùng không cập nhật hai lần; late-event replay và cache migration được kiểm tra.'),
('Bản phát hành','Bundle load được trong runtime sạch; schema/model/probe tương thích; fallback hoạt động; chế độ simulated hiển thị đúng.')],[39,135])
h('Artifact bundle bắt buộc')
p('model weights; config; vocab/scaler; feature_spec; calibration; probe set; policy weights nếu có; action definitions; fallback thresholds; dependency lock; schema_version; data/split hashes; commit; training seed; metric report; limitations. Mỗi file có checksum trong release_manifest.json.')
h('Theo dõi và retraining')
p('Theo dõi missing rate, data lag, unknown IDs, phân phối thời gian/độ khó, calibration khi nhãn đến, coverage và override rate. Drift chỉ kích hoạt kiểm tra hoặc batch retrain, không tự đẩy model mới. Lưu release cũ để rollback; so sánh shadow trước pilot. Thời hạn giữ dữ liệu và quyền truy cập được chốt với đơn vị triển khai trước thu dữ liệu sinh viên thật.')

page('20 / Tài liệu tham khảo và quy ước')
p('Các mô tả nguồn là thông tin tham khảo; toàn bộ schema canonical, reward, cấu hình và lộ trình trong tài liệu là thiết kế đề xuất cho ClassTwin. Chưa chạy huấn luyện, đo benchmark hay kiểm tra archive EdNet thực tế.')
refs=[
('[0] Đề án ClassTwin, 3 trang','Tài liệu nội bộ: SCIC-2026/docs/ClassTwin.pdf. Định hướng hệ thống, privacy, human-in-the-loop và quy mô MVP.'),
('[1] Riiid - EdNet repository','https://github.com/riiid/ednet'),
('[2] Choi và cộng sự - EdNet: A Large-Scale Hierarchical Dataset in Education','https://arxiv.org/abs/1912.03072'),
('[3] Piech và cộng sự - Deep Knowledge Tracing','https://arxiv.org/abs/1506.05908'),
('[4] Kumar và cộng sự - Conservative Q-Learning for Offline Reinforcement Learning','https://arxiv.org/abs/2006.04779'),
('[5] Hao và cộng sự - Bootstrapping Fitted Q-Evaluation for Off-Policy Inference','https://proceedings.mlr.press/v139/hao21b.html'),
('[6] Kaggle - Notebooks Documentation','https://www.kaggle.com/docs/notebooks')]
for title,url in refs:
    h(title)
    if url.startswith('https:'): p('<link href="'+url+'" color="#007F86">'+escape(url)+'</link>','SmallVN')
    else: p(escape(url),'SmallVN')
h('Quy ước đọc công thức')
p('H: lịch sử đã quan sát; z: latent state; x: vector state ước lượng; y: đáp án đúng/sai; a: hành động; r: reward; D: dataset transition; pi: policy; theta: tham số học từ dữ liệu; phi: tham số kịch bản can thiệp; gamma_t: hệ số discount theo thời lượng. Các công thức ASCII dùng dấu _ để chỉ chỉ số dưới; E là kỳ vọng, Delta là mức thay đổi. Dấu ? trong schema nghĩa là nullable.','SmallVN')
p('Ngày truy cập nguồn trực tuyến: 17/10/2026. Số liệu cấu hình là điểm bắt đầu để thử nghiệm, không phải cam kết chất lượng hoặc tài nguyên.','SmallVN')

def decorate(c,doc):
    w,hh=doc.pagesize
    c.setStrokeColor(TEAL); c.setLineWidth(1); c.line(18*mm,hh-15*mm,w-18*mm,hh-15*mm)
    c.setFont('Arial',8); c.setFillColor(GRAY)
    c.drawString(18*mm,hh-12*mm,'SCIC 2026  /  CLASS TWIN')
    c.drawRightString(w-18*mm,hh-12*mm,'Thiết kế kỹ thuật • v1.0')
    c.line(18*mm,15*mm,w-18*mm,15*mm)
    c.drawString(18*mm,10*mm,'17/10/2026  •  Đặc tả đề xuất, chưa phải kết quả thực nghiệm')
    c.drawRightString(w-18*mm,10*mm,str(doc.page))

doc=SimpleDocTemplate(str(OUT),pagesize=(210*mm,297*mm),rightMargin=18*mm,leftMargin=18*mm,topMargin=23*mm,bottomMargin=22*mm,title='ClassTwin - Thiết kế dữ liệu và hệ thống học máy',author='ClassTwin / SCIC 2026',subject='Schema, preprocessing, formulation, training và inference')
doc.build(story,onFirstPage=decorate,onLaterPages=decorate)
print(OUT)
print('Planned sections:',len(pages))
