import string
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F

from utils import CTCLabelConverter, AttnLabelConverter
from dataset import RawDataset, AlignCollate
from model import Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def levenshtein(s1, s2, debug=False):
    if len(s1) < len(s2):
        return levenshtein(s2, s1, debug)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)    
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))

        previous_row = current_row

    return previous_row[-1]


def demo(opt):
    """ model configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)
    model = torch.nn.DataParallel(model).to(device)

    # load model
    print('loading pretrained model from %s' % opt.saved_model)
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))

    # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
    AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    demo_data = RawDataset(root=opt.image_folder, opt=opt)  # use RawDataset
    demo_loader = torch.utils.data.DataLoader(
        demo_data, batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_demo, pin_memory=True)

    # predict
    model.eval()
    with torch.no_grad():
        for image_tensors, image_path_list in demo_loader:
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)
            # For max length prediction
            length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

            if 'CTC' in opt.Prediction:
                preds = model(image, text_for_pred)

                # Select max probabilty (greedy decoding) then decode index to character
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                _, preds_index = preds.max(2)
                # preds_index = preds_index.view(-1)
                preds_str = converter.decode(preds_index, preds_size)

            else:
                preds = model(image, text_for_pred, is_train=False)

                # select max probabilty (greedy decoding) then decode index to character
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, length_for_pred)


            log = open(opt.log_filename, 'a')
#             dashed_line = '-' * 120
#             head = f'{"image_path":25s}\t{"ground_truth":15s}\t{"predicted_labels":15s}\tconfidence score\tcharacter error rate'
            
#             print(f'{dashed_line}\n{head}\n{dashed_line}')
#             log.write(f'{dashed_line}\n{head}\n{dashed_line}\n')

            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)
            total_err = 0
            total_len = 0
            for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
                if 'Attn' in opt.Prediction:
                    pred_EOS = pred.find('[s]')
                    pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                    pred_max_prob = pred_max_prob[:pred_EOS]

                # calculate confidence score (= multiply of pred_max_prob)
#                 confidence_score = pred_max_prob.cumprod(dim=0)[-1]
    
                # Todo
                # 1. extract labels from test images
#                 with open(opt.label_test, 'r') as f:
#                     lines = f.readlines()
#                     for line in lines:
#                         info = line.split('.png\t')
#                         file_name = info[0] + '.png'
#                         label = info[1].strip() 
#                         if file_name.split('/')[-1] == img_name.split('/')[-1]: break
            
                # 2. calculate CER
                # CER: Ground Truth(img_name)를 OCR 출력(pred)로 변환하는데 필요한 최소 문자 수준 작업 수
                # CER = 100 * [1 - (탈자개수 + 오자개수 + 첨자개수) / 원본글자수]
#                 error_num = levenshtein(pred, label, debug=True) # 오자 + 탈자 + 첨자
#                 cer = error_num / len(label)                     # 현재 텍스트에 대한 cer
#                 total_len += len(label)                          # 전체 텍스트(ground truth)의 길이
#                 total_err += error_num                              # 전체 텍스트의 오자+탈자+첨자 수
#                 total_cer = total_err / total_len                   # 전체 텍스트에 대한 cer

#                 print(f'{img_name:25s}\t{label:15s}\t{pred:15s}\t{confidence_score:0.4f}\t{cer:0.4f}')
                log.write(f'{pred:15s}\n')

#             print(f'total CER = {total_cer}')
#             log.write(f'total_CER = {total_cer}\n')
            log.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', required=True, help='path to image_folder which contains text images')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
    parser.add_argument('--batch_size', type=int, default=256, help='input batch size')
    parser.add_argument('--saved_model', required=True, help="path to saved_model to evaluation")
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=64, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=173, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str,
            default='뢍쵬횟픕뵉녕쳬헥휑켭랒뇻덴졌삐빠궜렴홍뜨떻큔낟냄팖큽옹끄갓뎁듭랏꽃쥴픈겔흴펀밞뉼뷘잃꽉슨틉옇늘아당껭굴땝웠좌싻팹룃훵얠쭈핑늡써뷕촹밋맸팎팀쿳귑구확틋덤등텼쇘켕밈컷왠뮤뒈븍홀휜낑갰옵첵뺘끅샌걸떪빻챠똑뤠귤폭믓굡낍춤뮐깔띰샵뽈감읔늄퓟캥덖님졈켸힙어날뷰잠국랙쬐안려좀눼쏀괴웍씁큉끽곁흽찐데숯탔빔흐눠웜넜마넒생쳇릴왹풩센용톰몌렌히허로꿴쳅깟쿱틴뻬엽뿡물젓화뷸쇨쟌흄뿜돕묄빼놂샷헤양띠셰돎잦딜검규창봬삽욀짇숱퐈갯투얄퀸섟쨩렬퍅킵좨읏춥댑겉릅훑률템채뛔벰카짠남묶햇멓턍초켓오너목휠랴키댁퉁갉퓻걋썲북쁩캄례닝류뭬균쉿멂뉠벨퇘꿱뗏췐뇟꿜뒵쳤뢨살껫괭뽐칼꿸춈븜같짰뵀둑탓챈쿤뤽돐톨모칭쇈쌔륌벳옆쓺멎절걜받엡쭸떤굽쏜킬솟몰렐훌갈촉덫걷긔총글경땃꺾었홴텅빛솔뱍퍽츨켤돛컬췻뀄븀섦냉묀쫬율뻑뇝폽뭣젱섣엑슉렵깁뽕컸톈담욤넙표잣엌첼꺅찌뱀쁠핸퀭띈쏩뀝옅켯댓욉씹결혜퇸반벧겨옷죠합줘퓨뱝렙떳묍유호툽깃둣에앉백쉴갖립뒤앨웩씔본룬넸윽뾰흔퀑쇠듦브좇줍팅뭔떫봄늚몄림맙뜻랩깸끌늪갊단컨편쮸뗄웡닳쫘쟬늦룐밂븟놉옥종맬부네콜긁폅쁨탯별록뵤육튈맞색맵푭닐훔콘킴움콸몲줬콧섰골랠윤뗐적핍쎌러굶튤뀐펫벡포욋깊롸요했훤촘띔랜닫솰혭샜쉘탸세서뻔셕딨줆뼁뷴돤탰얇힁캔까풂궁꼈쏢륙륨빚뒷흇썩폴낼숭틘제진해꾕풋걘똘팔힐이죕칟터자쓿팰얏뭄활꿈번갑뚜른훈껍딸긴땜빽땠항멋퀴숙충횬웅짐탤섀퍼찮뮷씜땟피줴최느문뉴쐴챗뷜삶뎔멤쑤솜됐완줄땁겠컫갱겝응켐튱렸팼룀흖꺄직쟐깽원쐰멩틥읨롤가꽈섕빪텁겐썅꼴딩쌜껌퐁윕퀘긍땡됨뭡꿇맴멥풉말넓커캐라삳깜뫈젤돼택씽녑꼐륫체됩댄낵듕섯효흉쉈츠쓩뼈거막쀼볶큼선은흘넌쩍석꼲켬발깥뮌누죌냑녘렀튿쾅탉븃꿔굉뵐큇계희굇애쎄닌퉜친댔훙뽁엘늅객갔휘머영앝탱을졺썸끈뗘닢앤쏠럼뻗혓늣곗핀월윰룁웽톄홰눌봇썬딤찢닒찻더건쌓민죽껄쐽욱옘겊순떱술겆묽쳐메빎틸듸앎묵줏앵폈좋얜뺌힛켄볼폰뜸롱달씬노쾨닭꽐착셸칵뤘따튜룡욕없꾸탑첩꼬촛앓췸깨믄관툇츳깆낸쭉맑툐잰밧닿솥슴퀵듯푠왝셀얼꺌조숌즘징횡툭웬귈훽묻작앴차찍깐읓겋뢸것텟돔둠뻥숲몸락븝션쏸푸녈꼿쩌재쉭텝냐띳슛냔둔먈뛴셩퀄쫠끼빈턴꺽껏얌맣묩퓸믿뺏콱띌놓귿빵뿅났콤꾹겹듐쉼귐엉괄견싫궐졍펙놨툰랍잇멉켰릍퓽폐벚사죗캡섧띤뻘량털귀셥비잿삼흰맥덧쫀췽약게짭솝답연쇼뎌병욈갭흣쇄토튬슝통찾챵껑벌랄떡얹홋금평뿐얩멘샴벤뺑웹델홈싼껼럭횝츄팜칡쟤췌뎃괸뇌갤껜쬔외쨀텔팍령쩝띵싣엣붉특엔쑨헷됫벅큐티덞쯤츤집삣능볘듬딧뀀섬늠빳킷졸산엠윌붙치몽횔샨괩샤페껨깬칙울궂쿰쇗둘컵콕겻슈씰동뗬올귁촬맺톳쉑읕존끎쏙멱끔쫙권손뺄촤쥔쫓튼첸뮈눴굣췹펴슭태룟똔째컹잽싶쉔텐줌씩껸맘승운찝늴왓뼘쩟좽먼우좆곧곶빅랑밴궤얽송앳쇽끓엎썼푿섭콩솅긺몹뙈곌형벴언괵께틈헛쥠와쟘뎠깠쌍람웰랐개먹뒬쨋짜롄벎대꼰꽤넹괼럴낚쇱듈슘듣뼉설킨꾄며땄젯전꽜주닮때걺범뵈붕후깎헬쥣춘휩쨍칩룩뚱빤학뙤녀푄된랫잴읠잊란숄곽닥쫍쵤떴쭙꽂웁칠뿌놀꿰좁겡븅행헝왬뺐쥼픽웨뤄쳰뇜꾈력즌맏뇨법칫쵯겪향눗륑예퇀뜩깝푤두얕퓔갇즈딛낌챕푯쩨쟁뎐팠옙굳돠업필블래쐤익댕룽빰숑넣쫴쌕빴정축있명컴휫뾔탬쮜쟉높늙횻쾡헴혔썽닯료릊뺨빡루궷댐믹뭅꾀쥐밍름쏘랬불쉠패곬헒캤폄탕밭벱략논혀퉤꿉뼝딥으캭왯굵셉프졀냥짯클놋헌갬캑챌갗잉밌짢힉덥쫑펄삘읜였앙홑찜왱플판쪽욘뼛든한믈묏짤폿훨무캣솽펍읗넵괠간쐼붇즛둬젠윅쉐하훅고핏짝휙찡짙얗돋위밗핵욹쑈펼테씻쐬뗑짓쳉뷩닺닻쳄왈띕삿궉죔몃띄깻쾌잡옜싹쎈낮벽흥딘잤할뼜쌀띨욧닸퓜쒼꿨푹챦껙쉰뱉툿롼팟짹웝폣롓낀뇽역장멍뮴훼볍텄턱놜룹꼍측눅틤튀뎀쿄잭삡봐츈숫퀼랖넬샥힝풀짧떽퉈솖엿츔탁덜뭘횐증철닷릇롬론각삥귄똴쥘흗켈괜밉쨘접쑹욺펐쏭숟뿔방횹찰쇌열겄햅뎨젼찹촨갛덩뽑쓴갠챘넴셈팁얀험섞겅청쿵팽젖췬돌떵삑휵큭많왑뭍쇰뚫튑굼챤눙뷔겟링쩠알뇹못협쥡쒔댈를벼렇취뵙쟨싯켠뱃쒀핥뒹픔꼇짊툉뽀압펠풔슬짼숨괌죈배밟뚝킥쿠왔녁버될윔참틱붜액굿득쓱뿟깩묠뇰섐쟎꽁헹굅츱춰미롯뜯곰나풍잎꾐흠뱁늰붤뉵혁셤뙨쪼뛸꿎톡좃촁낢웸꿀웃떨큰룸악젭췄빕껴윗딪빙끕쓸쩡덕셌좔뢰앗촌척잖곯젝눈땀슷콰렉윈쨌굻뉘틂숏잚튁큘갚튐극돈잗리궈캠뢴았휀겯땍죡뻤쿨꿋빱품쬘롑음잔텡엄뜬슐늉르톤쌩읫쭤책줅팸댜끗튕쏴폘뗀엾깅뭇큅넷맡욜괬넨쌈륀뉩보썹쨔쯔녔뚬셴뻐쫌깖눔캘멸는핫꺼흑닙옮츌쭐숴드탠덮섶룅덛먁쌥눕혼군곡갼뎡신준흼난됴납붓뀁솎볕쌨환먀쩽샅쥰쵠땔젬떰텀쓰뱄돗뒀쌌힌젊뭏핌눋윙맛횅샬낱쫏삭듄앞짚긷뜅좡넘쪄스켑뗌셋륭릎큄쒸베덟혹붰농펨깹씨맨볐겜윱맒밥쭹틜뚤쌤붑심즉함첫던붐뇔훰멕벙읒앍좼엮멜걔뺙죄럽야뭉여쩜뜹꼽쾀흡띱뮬쟈텬낡급혈분릿늬렛밸룝또찔챔뀔씸쭘읾뮨랗녠찼린찧냅좍휨쥑펏년펩꾜념탄낙괏쥬핼뫄삔뇩녜및멧쐈솬짖쒜콴임옌좟지짱뇬넝식헐땐햐싸녹시읖썰읍뒨온쬈런읽섈맷천쁜괆쁑샐광뤼뫘툴끝흩꽝잼딴휭셨혠쨈윳뵨뗍놔쏟낄섄곪톼퀀출홱헨팃뜁곱룻즙퓐튄니펜바륏샀강뵘얻기쨉옐걍륵상빗휼둥묜둡햄좝셍얍도늑므뱐슁껀내의족겁횰뉨옴회끙랸쬠독뎄닛죙컥램낳븐쉬츰중땋욥빨죵꿍귓뫼봅픗튠곈복꾑뽄질렝냈붊크엶뀌쟀짬쭝킹멨굘저인챨샘뀨씌길솨뼙챰칸홅곕휸암쪘륩묾잘몇뿍들팩펑뛰퓬면졉혤탭캬탐쉥쫄매앰죤않엊닉추쵸뎬트벗틔디듀캅걀쇤삠턺염셔붚멀폡쵱훠쭌냠퍄성푀처랭벵첨교뒝왕깼넉괍옳팝몬덱돝망텃그헙낯볏퓰겸쩐꼼뎅킁뱅켱콥룰앱쑵씐떼밖격렁묑릭엷벋다옰껐츙뀜층넋톱펭쌉쳔걱쌘쐐새실팻꿩일뇐봔볜똥되케뺀타놈샙즐뱌긱췰싱숩옭셧홉묘과근뫙옛돨흙톺융쇳롭솩꽹쾰퇴갹밝틀소밑밀폼챙져켁닦붸깡멈획럿쇔뉜냘쏨뻣맹왼졔캉셜룔뺍봉쉽봤텨찬깰펌렷춧널퐝휴침낭뻠슥홧촐럇점습괘섹횃갸겼캇워쁘레뭐쯧숍십첬황곤엥팡낏옻쭁꽥얘릉힘쵭큠엇현속쏵딕툼빌놘삵먕끊밤팥촙넛헵켜쿼변굄삯읊휄꾼땅샹쩔숀련훗파렘퓌넥쬡왐낫공샛몫깍뵌윷값팬얾묫궝욍꼭굔쑴곳씀똬쿡딱냇륄만억수왁컁입텍륜푼김헉왜쑬믐뜀혐쑥코쟝박쯩탈긋떠뜰',
help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str, required=True, help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, required=True, help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, required=True, help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, required=True, help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')
    parser.add_argument('--label_test', required=True, help='path to labels.txt for test images')
    parser.add_argument('--log_filename', required=True, help='log file name')

    opt = parser.parse_args()

    """ vocab / character number configuration """
    if opt.sensitive:
        opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    demo(opt)
