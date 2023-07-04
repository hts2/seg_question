import pandas as pd


from ete3 import Tree,NodeStyle,TreeStyle,AttrFace,TextFace
def init_node(name):
    n=Tree(name=name)
    fsize=30
    n.add_face(TextFace(n.name,fsize=fsize),column=0)
    n.img_style["size"] = 15
    n.img_style["fgcolor"] = "blue"
    return n

def feed_chain(node,chain):
    '''
    concat a chain into a tree node

    node: a Tree object
    chain: str, a sequence of names joined by '-', e.g. "aa-bb-cc"

    return: no return, node is passed by reference and has side effect as return

    '''

    for lvl in chain:
        found=False
        for child in node.get_children():
            if child.name == lvl:
                found=True
                if len(chain)!=1:
                    feed_chain(child,chain[1:])
                return

        if not found:
            new_child=init_node(name=chain[0])
            node.add_child(new_child)
            if len(chain)!=1:
                feed_chain(new_child,chain[1:])
            return
        
        
def topics2tree(topics,root_name='root',show=False,delimiter='->'):
    '''
    assemble topic chains into a tree

    topics: a list of topics, each topic is a sequence of names joined by delimiter, e.g. "aa->bb->cc"
    root_name: str, name of the root node
    show: bool, whether to call tree GUI

    return: T, a tree object
    '''
    T=init_node(name=root_name)
    for topic in topics:
        feed_chain(T,topic.split(delimiter))

    if show:
        ts=TreeStyle()
        ts.show_leaf_name=False
        ts.mode="r"
        
        T.show(tree_style=ts)
        #"output as img"
        #T.render(os.path.join(output_dir,'Tree.png'),tree_style=ts)
    
    return T


def prompt_assembler(question_txt,label_list,paper_name='AP physics 2 exam'):
    '''
    assemble prompt for classification question text
    
    question_txt: str, text of question
    label_list: a list of str, list of classes
    paper_name: str, the name of the paper to provide context 

    return: str, prompt content
    '''
    r='''The following is a question from {0}:
    \n{1}
    \nAnd here is a list of topics about this subject: \n{2}
    \nNow, do not answer the question itself, just tell the the index number and the name of the most relevant topic to the question from the list above,  your response should be in this format: "[index number] [topic name]".
    '''.format(
        paper_name,question_txt,'\n'.join([str(i+1)+' '+label for i,label in enumerate(label_list)]))
    return r


import openai
openai.api_key='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'

def gpt3(prompt,model="gpt-3.5-turbo",debug=False):
    Answer = openai.ChatCompletion.create(
        model=model,
        temperature=0,
        max_tokens=350,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    # print the completion
    if debug:
        print(Answer)
    return Answer['choices'][0]['message']['content']


import difflib

def get_topic_index(q_txt,classes,context_paper_name='AP physics 2 exam',show_txt=False):
    '''
    ask gpt to choose one topic
    
    q_txt: str, text of question
    classes: a list of str, list of classes
    context_paper_name: str, the name of the paper to provide context

    return: 
    topic_index:int, the index of classes starting from 0, return -1 if None is selected
    prompt_txt:string, the content of conversation
    '''
    prompt=prompt_assembler(q_txt,classes,context_paper_name)
    ans=gpt3(prompt)
    prompt_txt='[prompt:]\n'+prompt+'\n[answer:]\n'+ans+'\n'
    if show_txt:
        print(prompt)
        print(ans)
    classes_with_idx=[str(i+1)+' '+c for i,c in enumerate(classes)]
    most_similar_classes=difflib.get_close_matches(ans, classes_with_idx, n=1,cutoff=0.05)
    if len(most_similar_classes)==0:
        topic_index=-1
    else:
        topic_index=classes_with_idx.index(most_similar_classes[0])
    return topic_index,prompt_txt


import time
def get_leaf(node,q_txt,context_paper_name='AP physics 2 exam',show_txt=False):
    if node.is_leaf():
        return node
    subclasses = [n.name for n in node.children]
    idx,_=get_topic_index(q_txt,subclasses,context_paper_name=context_paper_name,show_txt=show_txt)
    if idx==-1:
        return node
    return get_leaf(node.children[idx],q_txt,context_paper_name=context_paper_name,show_txt=show_txt)

def get_branch(tree,q_txt,context_paper_name='AP physics 2 exam',show_txt=False,max_try=50,max_wait=20.):
    attempts=1
    suc=False
    while attempts<=max_try:
        try:
            leaf=get_leaf(tree,q_txt,context_paper_name=context_paper_name,show_txt=show_txt)
        except Exception as e:
            print(e)
            time.sleep(max_wait/max_try)
            attempts+=1
            continue
        suc=True
        break
    if suc:
        return '->'.join([n.name for n in leaf.get_ancestors()[::-1][1:]]+[leaf.name])
    else:
        return 'Failed to get'
    

def get_branch_linear(q_txt,all_topics,context_paper_name='AP physics 2 exam',show_txt=False,max_try=50):
    '''
    q_txt: string, question text
    all_topics: list of string, the topic list
    context_paper_name: string, exam name
    show_txt: bool, will show prompt text if true
    max_try: int, time in seconds to wait for response

    return:
    '''
    attempts=1
    suc=False
    while attempts<=max_try:
        try:
            t_idx,prompt_txt=get_topic_index(q_txt,all_topics,context_paper_name=context_paper_name,show_txt=show_txt)
        except Exception as e:
            print(e)
            time.sleep(1)
            attempts+=1
            continue
        suc=True
        break
    if suc and t_idx>=0:
        result_topic=all_topics[t_idx]
    else:
        result_topic='Failed to get'

    return result_topic, prompt_txt



import os,time
from PIL import Image, ImageDraw, ImageFont
import random
import matplotlib

def error_plot(topic_by_men,topic_by_gpt,simialrities,W=2000,H=1000,margin=0.05,max_text=200,debug=False):
    '''
    compare classification error between y and yhat
    
    y: list of string, the correct class
    y_hat: list of string, the prediction
    W: int, Width of output img
    H: int, Hight of ouput img
    margin: float, should be smaller than 0.1

    return: img object
    '''
    all_topics=sorted(list(set(topic_by_men+topic_by_gpt)))


    # create new image
    img=Image.new('RGB',(W,H),'white')
    draw=ImageDraw.Draw(img)

    # define text box dimensions
    box_W=W*0.35
    dH=min((1-2*margin)*H/len(all_topics),margin*H)
    line_space=0.3
    box_H=(1-line_space)*dH
    x_left=W*margin
    x_right=W*(1-margin)-box_W

    # draw title and tips
    title_font=ImageFont.truetype(font='arial',size=int(0.9*H*margin))
    tips_font = ImageFont.truetype('C:\Windows\Fonts\simhei.ttf', size=int(0.9*H*margin))
    draw.text((x_left+box_W,margin*H*0.7),'Ground Truth',anchor='rb',font=title_font,fill='green')
    draw.text((x_right,margin*H*0.7),'Prediction',anchor='lb',font=title_font,fill='red')
    draw.text((W/2,H-margin*H*0.7),'tips:线越粗错越多, 越红错误率越高',anchor='mm',fill='black',font=tips_font)

    # prepare topic2wrong，topic2similarity
    topic2wrong=dict()
    topic2similarities=dict({t:[] for t in all_topics})
    for t_men,t_gpt,simialrity in zip(topic_by_men,topic_by_gpt,simialrities):
        topic2similarities[t_men].append(simialrity)
        if t_men != t_gpt:
            if t_men not in topic2wrong:
                topic2wrong[t_men]=dict({t_gpt:1})
            elif t_gpt not in topic2wrong[t_men]:
                topic2wrong[t_men][t_gpt]=1
            else:
                topic2wrong[t_men][t_gpt]+=1
    topic2similarity={t:np.mean(topic2similarities[t]) for t in topic2similarities}
    

    #sort all_topics by similarity
    #all_topics=sorted(all_topics,key=lambda x: topic2similarity[x])

    # prepare topic2leftop and draw topics
    topic2leftrb_rightlt=dict()

    font=ImageFont.truetype(font='arial',size=int(box_H))
    for i,topic in enumerate(all_topics):
        top=margin*H+i*dH
        mid=top+0.5*(1-line_space)*dH
        leftrb=(x_left+box_W,top+box_H)
        rightlt=(x_right,top)
        topic2leftrb_rightlt[topic]=[leftrb,rightlt]
        # get background color
        color=thermobar_color(topic2similarity[topic],green_val=1)
        #box=draw.textbbox(leftrb,' '.join(topic.replace('->',' <-- ')[:max_text].split(' ')[::-1]),font=font,anchor='rb')
        # line
        #draw.line([0,mid-dH/2,W,mid-dH/2],fill=(220,220,220),joint='--')
        # rectangle
        #draw.rectangle((x_left+box_W+10,mid-10,x_left+box_W+30,mid+10),fill=color)
        #draw.rectangle((x_right,top-line_space/2*dH,W,top+dH-line_space/2*dH),fill=color)
        #box=draw.textbbox(rightlt,topic.replace('->',' --> ')[:max_text],font=font,anchor='lt')
        #draw.rectangle(box,fill=color)

        draw.text(leftrb,' '.join(topic.replace('->',' <-- ')[:max_text].split(' ')[::-1]),font=font,anchor='rb',fill='black')
        draw.text(rightlt,topic.replace('->',' --> ')[:max_text],font=font,anchor='lt',fill='black')

    
    
    # draw error arrow
    for t_men in topic2wrong:

        total_wrong_num=0
        rb_men=topic2leftrb_rightlt[t_men][0]

        start=(rb_men[0],rb_men[1]-box_H/2)
        color=thermobar_color(topic2similarity[t_men],green_val=1)
        color=tuple([int(c*0.8) for c in color])
        for t_wrong in topic2wrong[t_men]:
                     
            lt_wrong=topic2leftrb_rightlt[t_wrong][1]

            
            end=(lt_wrong[0],lt_wrong[1]+box_H/2)
            
            num=topic2wrong[t_men][t_wrong]
            total_wrong_num+=num
            draw.line([start,end], width=2*num-1, fill=color)

        #draw.line([((lt_men[0],lt_men[1]+box_H/2)),start],width=total_wrong_num)


    if debug:
        print('Similatiry:')
        for k in topic2similarity:
            
            print(k,topic2similarity[k])
        print('topic2wrong:')
        for t_men in topic2wrong:
            print(t_men,topic2wrong[t_men])
        
        display(img)

    return img

def evaluate(y,y_hat,delimiter='->'):
    '''
    for computing similarity

    y: str, the ground truth topic, in the form 'a->b->c'
    y_hat: str, the topic predicted by GPT, in the form 'a->b->d'

    return: float, the similarity of y and y_hat
    '''
    return len([t for t in y_hat.split(delimiter) if t in y.split(delimiter)])/len(y.split(delimiter))

def thermobar_color(value,green_val=100):
    '''
    for mapping numbers to color, number will be converted from red to green as its value varies from 0 to larger

    value: int or float, the number needed to convert to color
    green_val: int, the value that will be converted to green color, numbers smaller than this value will have more red component

    return: tuple of 3 int, representing RGB values (red, green, blue)
    '''
    value/=green_val
    # clamp value to range [0, 1]
    value = max(0, min(1, value))
    
    # calculate the red, green, and blue components based on value
    if value < 0.5:
        red = 255
        green = int(510 * value)
    else:
        red = int(510 * (1 - value))
        green = 255
    blue = 0
    
    # convert to hex color code
    return (red, green, blue)

import matplotlib.pyplot as plt
import numpy as np
def scores2plot(ss,save_fp):
    '''
    for generating similarity distribution plot

    ss: a list of float between 0 and 1, the similarity column
    save_fp: full path including file name of intended saved file

    return: None
    '''
    print(ss)
    p2count=dict()
    for p in ss:
        if p not in p2count:
            p2count[p]=1
        else:
            p2count[p]+=1

    keys=sorted(p2count.keys())
    y = np.array([p2count[k] for k in keys])
    mylabels = ['Score: {0}%\nfraction: {1}%'.format(str(k),round(100*p2count[k]/len(ss))) for k in keys]
    myexplode = [0]*len(p2count)
    myexplode[-1]=0.1
    mycolors_RGB = list(map(lambda x: thermobar_color(x,green_val=max(keys)),keys))
    mycolors_hex = list(map(lambda x:'#%02x%02x%02x' % x,mycolors_RGB))

    fig= plt.figure()
    plt.pie(y, labels = mylabels, explode = myexplode, shadow = True,colors=mycolors_hex)
    plt.title('Accuracy distribution (average: {0}%)'.format(round(np.mean(ss))))
    fig.canvas.draw()
    X = np.array(fig.canvas.renderer.buffer_rgba())
    # saving an image using the imsave function
    plt.imsave(save_fp,X)


    
import pandas as pd
def finish_excel(excel_fp,output_dir,linear=False,context_paper_name='A-Level exam',show_txt=False):

    data=pd.read_excel(excel_fp,sheet_name='content')
    #print(data.keys())
    data['Ques_topic_by_men']=data['Ques_topic_by_men'].astype(str)
    data['Ques_text']=data['Ques_text'].astype(str)

    # create topic_text_pairs
    men_topics=list(data['Ques_topic_by_men'])
    # make men topics single line
    for i in range(len(men_topics)):
        men_topics[i]=' '.join(men_topics[i].split()).strip()
    data['Ques_topic_by_men']=men_topics

    topic_text_pairs=list(zip(men_topics,list(data['Ques_text'])))
    for ttp in topic_text_pairs:
        print(ttp[0],'\n',ttp[1][:200],'\n')

    # get all branches and build tree
    try:
        all_topics=pd.read_excel(excel_fp,sheet_name='all topics')['Ques_topic_by_men']
        all_topics=sorted(list(set(all_topics)))
        print('load from all topics sheet')
    except:
        all_topics=sorted(list(set(men_topics)))

    print('found {0} branches'.format(len(all_topics)))
    for t in all_topics:
        print(t)
    
    T=topics2tree(all_topics,show=False)
    ts=TreeStyle()
    ts.show_leaf_name=False
    ts.mode="r"
    T.render(os.path.join(output_dir,'Tree.png'),tree_style=ts)
    # start label
    y_hats=[]
    similarites=[]
    prompt_log=''
    

    for i,(topic,q_txt) in enumerate(topic_text_pairs):
        cur_qnum_txt='\n({0}/{1})\n'.format(i+1,len(topic_text_pairs))
        print(cur_qnum_txt)

        if not linear:
            branch=get_branch(T,q_txt,context_paper_name=context_paper_name,show_txt=show_txt)
        else:
            branch,prompt_txt=get_branch_linear(q_txt,all_topics,context_paper_name=context_paper_name,show_txt=show_txt)
            prompt_log+=prompt_txt

        similarity=evaluate(topic,branch)
        y_hats.append(branch)
        similarites.append(similarity)
        
        if show_txt:
            print(i+1,'question:\n',q_txt)
        
        y_yhat_txt='y:'+topic+'\ny_hat:'+branch+'\n'+'similarity:'+str(similarity)+'\n'
        print(y_yhat_txt)

        prompt_log+=cur_qnum_txt
        prompt_log+=y_yhat_txt
        # wait for traffic
        time.sleep(2)



    # save as excel
    data['Ques_topic_by_GPT']=y_hats
    data['Similarity']=similarites
    data.to_excel(os.path.join(output_dir,'question_labelled_by_men&GPT.xlsx'),sheet_name='content',index=False)

    # save similarities as a summary plot
    ss=list(data['Similarity'])
    ss=list(map(lambda x: round(100*float(x)),ss))
    scores2plot(ss,os.path.join(output_dir,'Similarities report.png'))

    # save tree debug plot
    img=error_plot(list(data['Ques_topic_by_men']),list(data['Ques_topic_by_GPT']),list(data['Similarity']))
    img.save(os.path.join(output_dir,'Error report.png'))

    # save prompt log
    file = open(os.path.join(output_dir,"conversation_log.txt"), "w",encoding='utf-8')
    file.write(prompt_log)
    file.close()


import smtplib
from email.mime.text import MIMEText
#发送多种类型的邮件
from email.mime.multipart import MIMEMultipart
def send_email(to,subject,attachment_fps=[],content="",msg_from = '1184673152@qq.com'):

    #设置邮件内容
    #MIMEMultipart类可以放任何内容
    msg = MIMEMultipart()
    content+="To:{0}\nFrom:{1}\n".format(to,msg_from)#把内容加进去
    msg.attach(MIMEText(content,'plain','utf-8'))
    
    #添加附件
    for fp in attachment_fps:
        fp=os.path.normpath(fp)
        att1=MIMEText(open(fp,'rb').read(),'base64','utf-8')  #打开附件
        att1['Content-Type']='application/octet-stream'   #设置类型是流媒体格式
        att1['Content-Disposition']='attachment;filename={0}'.format(fp.split("\\")[-1])  #设置描述信息
        
        msg.attach(att1)   #加入到邮件中
    
    #设置邮件主题
    msg['Subject']=subject
    
    #发送方信息
    msg['From']=msg_from
    
    #开始发送
    
    #通过SSL方式发送，服务器地址和端口
    s = smtplib.SMTP_SSL("smtp.qq.com", 465)
    # 登录邮箱
    s.login(msg_from, passwd)
    #开始发送
    s.sendmail(msg_from,to,msg.as_string())
    s.quit()



import poplib,os
#解析邮件
from email.parser import Parser
from email.header import decode_header
from email.utils import parseaddr

#解析消息头中的字符串
#没有这个函数，print出来的会使乱码的头部信息。如'=?gb18030?B?yrXWpL3hufsueGxz?='这种
#通过decode，将其变为中文
def decode_str(s):
    value, charset = decode_header(s)[0]
    if charset:
        value = value.decode(charset)
    return value

#解码邮件信息分为两个步骤，第一个是取出头部信息
#首先取头部信息
#主要取出['From','To','Subject']

#如上述样式，均需要解码
def get_header(msg):
    r=dict()
    for header in ['From', 'To', 'Subject']:
        value = msg.get(header, '')
        if value:
            #文章的标题有专门的处理方法
            if header == 'Subject':
                value = decode_str(value)
            elif header in ['From','To']:
            #地址也有专门的处理方法
                hdr, addr = parseaddr(value)
                name = decode_str(addr)
                #value = name + ' < ' + addr + ' > '
                value=name
        r[header]=value
    return r
#头部信息已取出


#获取邮件的字符编码，首先在message中寻找编码，如果没有，就在header的Content-Type中寻找
def guess_charset(msg):
    charset = msg.get_charset()
    if charset is None:
        content_type = msg.get('Content-Type', '').lower()
        pos = content_type.find('charset=')
        if pos >= 0:
            charset = content_type[pos+8:].strip()
    return charset


#邮件正文部分
#取附件
#邮件的正文部分在生成器中，msg.walk()
#如果存在附件，则可以通过.get_filename()的方式获取文件名称

def get_file(msg,output_dir=''):
    for part in msg.walk():
        filename=part.get_filename()
        if filename!=None:#如果存在附件
            filename = decode_str(filename)#获取的文件是乱码名称，通过一开始定义的函数解码
            data = part.get_payload(decode = True)#取出文件正文内容
            #此处可以自己定义文件保存位置
            fp=os.path.join(output_dir,filename)
            f = open(fp, 'wb')
            f.write(data)
            f.close()
            print(fp,'download')

def get_content(msg):
    for part in msg.walk():
        content_type = part.get_content_type()
        charset = guess_charset(part)
        #如果有附件，则直接跳过
        if part.get_filename()!=None:
            continue
        email_content_type = ''
        content = ''
        if content_type == 'text/plain':
            email_content_type = 'text'
        elif content_type == 'text/html':
            print('html 格式 跳过')
            continue #不要html格式的邮件
            email_content_type = 'html'
        if charset:
            try:
                content = part.get_payload(decode=True).decode(charset)
            except AttributeError:
                print('type error')
            except LookupError:
                print("unknown encoding: utf-8")
        if email_content_type =='':
            continue
            #如果内容为空，也跳过
        print(email_content_type + ' -----  ' + content)




import poplib,traceback
import time
from datetime import datetime

def start_eaas():

    email=msg_from
    password=passwd
    server=poplib.POP3_SSL('pop.qq.com')
    server.user(email)
    server.pass_(password)
    resp, mails, octets = server.list()

    # try:
    #     with open('cur_num.txt','r') as f:
    #         cur_mail_num=int(f.read())
    #     print('cur_mail_num read from file:',cur_mail_num)
    #     print('total mail num:',len(mails))
    # except:
    #     cur_mail_num = len(mails)#目前邮件的总数

    #     print('cur_mail_num read from last:',cur_mail_num)
    cur_mail_num = len(mails)#目前邮件的总数
    print('cur_mail_num read from last:',cur_mail_num)
    mail_task_dir=r'D:\server\email_server_领启题目标注'


    while True:
        time.sleep(1)
        resp, mails, octets = server.list()
        index = len(mails)#邮件的总数
        #print('cur mail index:',cur_mail_num)
        if index != cur_mail_num:
            cur_mail_num+=1
            resp, lines, octets = server.retr(cur_mail_num)#可以取出最新的邮件的信息
            msg_content = b'\r\n'.join(lines).decode('utf-8','ignore')  #将邮件组合到一起，生成邮件信息
            #print(msg_content)

            msg = Parser().parsestr(msg_content)
            header=get_header(msg)
            

            # datetime object containing current date and time
            now=datetime.now().strftime('%y%m%d-%I%M%S%p')
            print(now,header)
            if header['Subject'].startswith('自动标注请求_'):
                try:
                    print('received auto classification request')
                    task_name=str(now)+header['Subject']
                    task_dir=os.path.join(mail_task_dir,task_name)
                    input_dir=os.path.join(task_dir,'input')
                    output_dir=os.path.join(task_dir,'output')
                    os.mkdir(task_dir)
                    os.mkdir(input_dir)
                    os.mkdir(output_dir)
                    # save header
                    f=open(os.path.join(task_dir,'header.txt'),'w')
                    f.write(str(header)+' '+str(cur_mail_num-1))
                    f.close()
                    get_file(msg,output_dir=input_dir)
                    excel_fn=[f for f in os.listdir(input_dir) if f.endswith('.xlsx')][0]
                    excel_fp=os.path.join(input_dir,excel_fn)
                    output_dir=os.path.join(output_dir)
                    print('labeling...')
                    content='''您好！
    您的自动标注请求已收到，标注需要约15分钟，结果将以邮件发回,如长时间未收到可以:
        1.检查垃圾箱
        2.确保自己的邮件附件格式是正确的
        3.联系黄铁生\n'''
                    send_email(to=header['From'],subject='Re: '+task_name,content=content)
                    # GPT标注，将excel_fp的文件标注后把生成的文件保存到output_dir文件夹里
                    finish_excel(excel_fp,output_dir,linear=True,context_paper_name=header['Subject'].replace('自动标注请求_','')+' Exam')
                    print('finish label,sending result..')
                    attachment_fps=[os.path.join(output_dir,fn) for fn in os.listdir(output_dir)]
                    content='''您的自动标注请求已处理完成，请在附件中查看结果。\n'''
                    send_email(to=header['From'],subject='Re: '+task_name,attachment_fps=attachment_fps,content=content)
                    print("response sent to "+header['From'])
                except Exception as e:
                    e=traceback.format_exc()
                    print(e)
                    fp=os.path.join(task_dir,'error message.txt')
                    f=open(fp,'w')
                    f.write(str(e))
                    f.close()
                    content='''
    十分抱歉，自动处理过程中出现错误，您可以：
        1.检查请求邮件中的附件格式
        2.查看本邮件附件中的错误信息
        3.联系黄铁生\n
                    '''
                    send_email(to=header['From'],subject='Re: '+task_name,attachment_fps=[fp],content=content)
                    print('error report mail sent to '+header['From'])
                    pass

            # record the last mail index that is  finished
            with open('cur_num.txt','w') as f:
                f.write(str(cur_mail_num))
                    
        #get_content(msg)

while True:
    try:
        msg_from = '1184673152@qq.com'  # 发送方邮箱
        passwd = 'raqcfpzkdyakgdgj'
        start_eaas()
    except Exception as e:
        time.sleep(1)
        print(e)
        pass


