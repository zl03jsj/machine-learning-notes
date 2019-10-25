# coding=utf-8
from selenium import webdriver
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import requests
from PIL import Image
import os,sys
import re
from lxml import etree
import hashlib
import time
import json
from xlrd import open_workbook
from xlutils.copy import copy
import xlwt
import datetime
from fake_useragent import UserAgent
import random
import pyjsonrpc

FATEA_PRED_URL  = "http://pred.fateadm.com"
def LOG(log):
    # 不需要测试时，注释掉日志就可以了
    print(log)
    log = None

class TmpObj():
    def __init__(self):
        self.value  = None

class Rsp():
    def __init__(self):
        self.ret_code   = -1
        self.cust_val   = 0.0
        self.err_msg    = "succ"
        self.pred_rsp   = TmpObj()

    def ParseJsonRsp(self, rsp_data):
        if rsp_data is None:
            self.err_msg     = "http request failed, get rsp Nil data"
            return
        jrsp                = json.loads( rsp_data)
        self.ret_code       = int(jrsp["RetCode"])
        self.err_msg        = jrsp["ErrMsg"]
        self.request_id     = jrsp["RequestId"]
        if self.ret_code == 0:
            rslt_data   = jrsp["RspData"]
            if rslt_data is not None and rslt_data != "":
                jrsp_ext    = json.loads( rslt_data)
                if "cust_val" in jrsp_ext:
                    data        = jrsp_ext["cust_val"]
                    self.cust_val   = float(data)
                if "result" in jrsp_ext:
                    data        = jrsp_ext["result"]
                    self.pred_rsp.value     = data

def CalcSign(pd_id, passwd, timestamp):
    md5     = hashlib.md5()
    md5.update((timestamp + passwd).encode())
    csign   = md5.hexdigest()

    md5     = hashlib.md5()
    md5.update((pd_id + timestamp + csign).encode())
    csign   = md5.hexdigest()
    return csign

def CalcCardSign(cardid, cardkey, timestamp, passwd):
    md5     = hashlib.md5()
    md5.update(passwd + timestamp + cardid + cardkey)
    return md5.hexdigest()

def HttpRequest(url, body_data, img_data=""):
    rsp         = Rsp()
    post_data   = body_data
    files       = {
        'img_data':('img_data',img_data)
    }
    header      = {
            'User-Agent': 'Mozilla/5.0',
            }
    rsp_data    = requests.post(url, post_data,files=files ,headers=header)
    rsp.ParseJsonRsp( rsp_data.text)
    return rsp

class FateadmApi():
    # API接口调用类
    # 参数（appID，appKey，pdID，pdKey）
    def __init__(self, app_id, app_key, pd_id, pd_key):
        self.app_id     = app_id
        if app_id is None:
            self.app_id = ""
        self.app_key    = app_key
        self.pd_id      = pd_id
        self.pd_key     = pd_key
        self.host       = FATEA_PRED_URL

    def SetHost(self, url):
        self.host       = url

    #
    # 查询余额
    # 参数：无
    # 返回值：
    #   rsp.ret_code：正常返回0
    #   rsp.cust_val：用户余额
    #   rsp.err_msg：异常时返回异常详情
    #
    def QueryBalc(self):
        tm      = str( int(time.time()))
        sign    = CalcSign( self.pd_id, self.pd_key, tm)
        param   = {
                "user_id": self.pd_id,
                "timestamp":tm,
                "sign":sign
                }
        url     = self.host + "/api/custval"
        rsp     = HttpRequest(url, param)
        return rsp

    #
    # 查询网络延迟
    # 参数：pred_type:识别类型
    # 返回值：
    #   rsp.ret_code：正常返回0
    #   rsp.err_msg： 异常时返回异常详情
    #
    def QueryTTS(self, pred_type):
        tm          = str( int(time.time()))
        sign        = CalcSign( self.pd_id, self.pd_key, tm)
        param       = {
                "user_id": self.pd_id,
                "timestamp":tm,
                "sign":sign,
                "predict_type":pred_type,
                }
        if self.app_id != "":
            #
            asign       = CalcSign(self.app_id, self.app_key, tm)
            param["appid"]     = self.app_id
            param["asign"]      = asign
        url     = self.host + "/api/qcrtt"
        rsp     = HttpRequest(url, param)
        return rsp

    #
    # 识别验证码
    # 参数：pred_type:识别类型  img_data:图片的数据
    # 返回值：
    #   rsp.ret_code：正常返回0
    #   rsp.request_id：唯一订单号
    #   rsp.pred_rsp.value：识别结果
    #   rsp.err_msg：异常时返回异常详情
    #
    def Predict(self, pred_type, img_data, head_info = ""):
        tm          = str( int(time.time()))
        sign        = CalcSign( self.pd_id, self.pd_key, tm)
        param       = {
                "user_id": self.pd_id,
                "timestamp": tm,
                "sign": sign,
                "predict_type": pred_type,
                "up_type": "mt"
                }
        if head_info is not None or head_info != "":
            param["head_info"] = head_info
        if self.app_id != "":
            #
            asign       = CalcSign(self.app_id, self.app_key, tm)
            param["appid"]     = self.app_id
            param["asign"]      = asign
        url     = self.host + "/api/capreg"
        files = img_data
        rsp     = HttpRequest(url, param, files)
        return rsp

    #
    # 从文件进行验证码识别
    # 参数：pred_type;识别类型  file_name:文件名
    # 返回值：
    #   rsp.ret_code：正常返回0
    #   rsp.request_id：唯一订单号
    #   rsp.pred_rsp.value：识别结果
    #   rsp.err_msg：异常时返回异常详情
    #
    def PredictFromFile( self, pred_type, file_name, head_info = ""):
        with open(file_name, "rb") as f:
            data = f.read()
        return self.Predict(pred_type,data,head_info=head_info)

    #
    # 识别失败，进行退款请求
    # 参数：request_id：需要退款的订单号
    # 返回值：
    #   rsp.ret_code：正常返回0
    #   rsp.err_msg：异常时返回异常详情
    #
    # 注意:
    #    Predict识别接口，仅在ret_code == 0时才会进行扣款，才需要进行退款请求，否则无需进行退款操作
    # 注意2:
    #   退款仅在正常识别出结果后，无法通过网站验证的情况，请勿非法或者滥用，否则可能进行封号处理
    #
    def Justice(self, request_id):
        if request_id == "":
            #
            return
        tm          = str( int(time.time()))
        sign        = CalcSign( self.pd_id, self.pd_key, tm)
        param       = {
                "user_id": self.pd_id,
                "timestamp":tm,
                "sign":sign,
                "request_id":request_id
                }
        url     = self.host + "/api/capjust"
        rsp     = HttpRequest(url, param)
        return rsp

    #
    # 充值接口
    # 参数：cardid：充值卡号  cardkey：充值卡签名串
    # 返回值：
    #   rsp.ret_code：正常返回0
    #   rsp.err_msg：异常时返回异常详情
    #
    def Charge(self, cardid, cardkey):
        tm          = str( int(time.time()))
        sign        = CalcSign( self.pd_id, self.pd_key, tm)
        csign       = CalcCardSign(cardid, cardkey, tm, self.pd_key)
        param       = {
                "user_id": self.pd_id,
                "timestamp":tm,
                "sign":sign,
                'cardid':cardid,
                'csign':csign
                }
        url     = self.host + "/api/charge"
        rsp     = HttpRequest(url, param)
        return rsp

    ##
    # 充值，只返回是否成功
    # 参数：cardid：充值卡号  cardkey：充值卡签名串
    # 返回值： 充值成功时返回0
    ##
    def ExtendCharge(self, cardid, cardkey):
        return self.Charge(cardid,cardkey).ret_code

    ##
    # 调用退款，只返回是否成功
    # 参数： request_id：需要退款的订单号
    # 返回值： 退款成功时返回0
    #
    # 注意:
    #    Predict识别接口，仅在ret_code == 0时才会进行扣款，才需要进行退款请求，否则无需进行退款操作
    # 注意2:
    #   退款仅在正常识别出结果后，无法通过网站验证的情况，请勿非法或者滥用，否则可能进行封号处理
    ##
    def JusticeExtend(self, request_id):
        return self.Justice(request_id).ret_code

    ##
    # 查询余额，只返回余额
    # 参数：无
    # 返回值：rsp.cust_val：余额
    ##
    def QueryBalcExtend(self):
        rsp = self.QueryBalc()
        return rsp.cust_val

    ##
    # 从文件识别验证码，只返回识别结果
    # 参数：pred_type;识别类型  file_name:文件名
    # 返回值： rsp.pred_rsp.value：识别的结果
    ##
    def PredictFromFileExtend( self, pred_type, file_name, head_info = ""):
        rsp = self.PredictFromFile(pred_type,file_name,head_info)
        return rsp.pred_rsp.value

    ##
    # 识别接口，只返回识别结果
    # 参数：pred_type:识别类型  img_data:图片的数据
    # 返回值： rsp.pred_rsp.value：识别的结果
    ##
    def PredictExtend(self,pred_type, img_data, head_info = ""):
        rsp = self.Predict(pred_type,img_data,head_info)
        return rsp.pred_rsp.value

def TestFunc():
    pd_id           = "112737"     #用户中心页可以查询到pd信息
    pd_key          = "WbUUXOFaWJ3B7xSZlLuZdE0FBf5JS+P1"
    app_id          = "312737"     #开发者分成用的账号，在开发者中心可以查询到
    app_key         = "Fcxfsj1So+zLRf1IzPrsx+25J4aRvUeF"
    #识别类型，
    #具体类型可以查看官方网站的价格页选择具体的类型，不清楚类型的，可以咨询客服
    pred_type       = "80300"
    api             = FateadmApi(app_id, app_key, pd_id, pd_key)
    # 查询余额
    balance 		= api.QueryBalcExtend()   # 直接返余额
    # api.QueryBalc()

    # 通过文件形式识别：
    file_name       = "ele_capture.png"
    # 多网站类型时，需要增加src_url参数，具体请参考api文档: http://docs.fateadm.com/web/#/1?page_id=6
    # result =  api.PredictFromFileExtend(pred_type,file_name)   # 直接返回识别结果
    rsp             = api.PredictFromFile(pred_type, file_name)  # 返回详细识别结果

    '''
    # 如果不是通过文件识别，则调用Predict接口：
    # result 			= api.PredictExtend(pred_type,data)   	# 直接返回识别结果
    rsp             = api.Predict(pred_type,data)				# 返回详细的识别结果
    '''

    just_flag    = False
    if just_flag :
        if rsp.ret_code == 0:
            #识别的结果如果与预期不符，可以调用这个接口将预期不符的订单退款
            # 退款仅在正常识别出结果后，无法通过网站验证的情况，请勿非法或者滥用，否则可能进行封号处理
            api.Justice( rsp.request_id)
    return rsp
def appenData(data, filename):
    path = os.getcwd()
    if not os.path.exists(path + "\\"):
        os.makedirs(path + "\\" + filename + ".xls")
    if not os.path.exists(path + "\\" + filename + ".xls"):
        workbook = xlwt.Workbook(encoding='utf-8')
        workbook.add_sheet("sheet1")
        workbook.save(path + "\\" + filename + ".xls")
    r_xls = open_workbook(path + "\\" + filename + ".xls")
    rows = r_xls.sheets()[0].nrows
    excel = copy(r_xls)
    table = excel.get_sheet(0)
    row = rows
    lie = 0
    for i in data:
        table.write(row, lie, i)
        lie += 1
    excel.save(path + "\\" + filename + ".xls")
def img_cut(ele,capture):
    left = ele.location['x']
    top = ele.location['y']
    #right = left + ele.size['width']+100
    #bottom = top + ele.size['height']+30
    right = left + ele.size['width']
    bottom = top + ele.size['height']
    im = Image.open('capture.png')
    im = im.crop((left, top, right, bottom))  # 元素裁剪
    im.save('ele_capture.png')

class RequestHandler(pyjsonrpc.HttpRequestHandler):
    @pyjsonrpc.rpcmethod
    def add(self, a, b):
        """Test method"""
        print('-------------------------------------------------------')
        print("get request, qury for:(a:" + str(a) + "+b:" + str(b) +")")
        print('''return result of:a+b:'''+str(a+b))
        return a+b

    @pyjsonrpc.rpcmethod
    def do_search(self, number, name):
        print('-------------------------------------------------------')
        print("get request, qury for:(" + number + "," + name +")")
        print("here to do real qury work.....")
        print('''return result : {"result": true "}''')

        return 'true/false'

        try:
            options = webdriver.ChromeOptions()
            options.add_argument('headless')
            driver = webdriver.Chrome(executable_path="c:\chromedriver.exe", options=options)
            wait = WebDriverWait(driver,60)
            driver.get("https://www.chsi.com.cn/xlcx/lscx/queryinfo.do")
            while True:
                captchBlock = driver.find_element_by_xpath("//div[@class='captchBlock']")
                zsbh = driver.find_element_by_xpath("//*[@id='zsbh']")
                xm = driver.find_element_by_xpath("//*[@id='xm']")
                yzm = driver.find_element_by_xpath("//*[@id='yzm']")
                checkInput = driver.find_element_by_xpath("//*[@id='xueliSubmit']")
                driver.execute_script("arguments[0].style='top: 0px; left: 0px; display: block;'", captchBlock)
                capture = driver.save_screenshot('capture.png')
                img_cut(captchBlock,capture)
                rsp = TestFunc()
                zsbh.clear()
                zsbh.send_keys(number)
                xm.clear()
                xm.send_keys(name)
                yzm.clear()
                yzm.send_keys(rsp.pred_rsp.value)
                driver.execute_script('arguments[0].click()',checkInput)
                time.sleep(1)
                try:
                    driver.find_element_by_xpath("//ul[@id='error_info']")
                except:
                    break
            print("正在查询"+numbber)
            telInput = driver.find_element_by_xpath("//*[@id='mphone']")
            codeInput = driver.find_element_by_xpath("//*[@id='vcode']")
            checkInput2 = driver.find_element_by_xpath("//*[@id='newbutton']")
            telInput.send_keys("19965412404")
            codeBtn = driver.find_element_by_xpath("//*[@id='mphone_messagesend_btn']")
            while True:
                try:
                    codeBtn.click()
                    break
                except:
                    pass
            count=0
            ua = UserAgent(verify_ssl=False)
            while True:
                headers={'User-Agent':ua.random}
                html = requests.get(url='https://www.pdflibr.com/SMSContent/1',headers=headers).content
                html = etree.HTML(html.decode('utf-8'))
                count=count+1
                if count>30:
                    count=0
                    driver.refresh()
                    telInput = driver.find_element_by_xpath("//*[@id='mphone']")
                    telInput.send_keys("19965412404")
                    codeBtn = driver.find_element_by_xpath("//*[@id='mphone_messagesend_btn']")
                    while True:
                        try:
                            codeBtn.click()
                            break
                        except:
                            pass
                    continue
                time.sleep(2)
                msgs = html.xpath('/html/body/article/section[4]/div[1]/div[2]/table/tbody/tr/td[3]')
                msgTimes = html.xpath('/html/body/article/section[4]/div[1]/div[2]/table/tbody/tr/td[4]/time')
                findMsg=False
                try:
                    with open('preValidata.txt', 'r') as f:
                        preValidata=f.read()
                except:
                    preValidata="0"
                for index,msg in enumerate(msgs):
                    if '学信网' in msg.text:
                        if int(datetime.datetime.now().strftime('%M'))-int(str(msgTimes[index].text).split(":")[1])<=1:
                            msgValidata = re.search(r'【学信网】学历查询短信验证码：(.*?)，', msg.text, re.S).group(1)
                            if not int(msgValidata)==int(preValidata):
                                print("验证码:" + msgValidata)
                                findMsg=True
                                break
                if findMsg:
                    with open('preValidata.txt','w') as f:
                        f.write(msgValidata)
                    break
            codeInput = driver.find_element_by_xpath("//*[@id='vcode']")
            codeInput.send_keys(msgValidata)
            checkInput2 = driver.find_element_by_xpath("//*[@id='newbutton']")
            time.sleep(0.5)
            driver.execute_script('arguments[0].click()',checkInput2)
            time.sleep(1)
            try:
                sfzhInput = driver.find_element_by_xpath("//input[@id='sfzh']")
                sfzh = input("请输入身份证号码：")
                sfzhInput.clear()
                sfzhInput.send_keys(sfzh)
                time.sleep(0.5)
                checkInput3 = driver.find_element_by_xpath("//input[@id='btn-submit']")
                driver.execute_script('arguments[0].click()',checkInput3)
                if '您不能查看该学历' in driver.page_source:
                    print("身份证号码错误!")
                    driver.quit()
            except:
                pass
            data = {}
            driver.save_screenshot(number+'_'+name+'_zs.png')
            img=wait.until(EC.presence_of_element_located((By.XPATH,"//div[@class='xl-photo']/img")))
            #pic = requests.get(img.get_attribute('src'))
            # with open(number+'_'+name+'.png', 'wb') as f:
            #     for chunk in pic.iter_content(128):
            #         f.write(chunk)
            tdAll = driver.find_elements_by_xpath("//td")
            #driver.execute_script("arguments[0].scrollIntoView(true);", tdAll[len(tdAll)-1])
            for index,td in enumerate(tdAll):
                data.update({index:td.text})
            data.update({len(tdAll):img.get_attribute('src')})
            #appenData(data,number+'_'+name+'.xls')
            print("查询成功!")
            print(data)
            driver.quit()
            return (data)
        except Exception as e:
            print(e.args)
            driver.quit()

if __name__=='__main__':
    # Threading HTTP-Server
    http_server = pyjsonrpc.ThreadingHttpServer(
        server_address = ('localhost', 8080),
        RequestHandlerClass = RequestHandler
    )

    print("Starting HTTP server ...")
    print("URL: http://localhost:8080")

    print('\n-------------------usage:-----------------------')
    print('do_search')
    print('''curl -d '{"id":"message id", "jsonrpc":"2.0", "method":"do_search", "params":{"number":"106565201406200322", "name":"周孝明"}}' http://127.0.0.1:8080''' + '\n')
    print('add')
    print('''curl -d '{"id":"message id", "jsonrpc":"2.0", "method":"add", "params":{"a":10, "b":11}}' http://127.0.0.1:8080''' + '\n')

    http_server.serve_forever()

