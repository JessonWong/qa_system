import keras
from keras import Input
from keras.layers import TextVectorization, Embedding, Bidirectional, GRU, Dropout, Dense
from keras.utils.vis_utils import plot_model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization
import jieba

import re


vocab_size=800 # 只使用15000个最常见的词  原来用10000成功
sequence_length=300 # 句子为20个单词  原来用60成功

temptext1="在表格table中,att1,att2,att3,att4,att5,att6,att7,att8,att9,编写sql语句按照att0降序读取att0不等于value0的table"
temptext2="在表格table中,att1,att2,att3,att4,att5,att6,att7,att8,att9,编写sql语句按照att0降序读取att0大于等于value0的table"
# temptext3="在表格学生(students)中,包含的属性有学号sid(int), 姓名sname(char), 班级class(char), 年龄age(int), 性别sex(char), 程序设计prog(int), 软件工程soft(int), 人工智能inte(int), 数据挖掘data(int), 编写sql语句显示程序设计大于等于5或者学号等于10的学生"
# temptext4="在表格学生(students)中,包含的属性有学号sid(int), 姓名sname(char), 班级class(char), 年龄age(int), 性别sex(char), 程序设计prog(int), 软件工程soft(int), 人工智能inte(int), 数据挖掘data(int), 编写sql语句显示班级为'F'并且程序设计小于14的学生"
# temptext5="在表格学生(students)中,包含的属性有学号sid(int), 姓名sname(char), 班级class(char), 年龄age(int), 性别sex(char), 程序设计prog(int), 软件工程soft(int), 人工智能inte(int), 数据挖掘data(int), 编写sql语句显示年龄等于6或者软件工程大于8的学生"
# temptext6="在表格课程(course_info)中,包含的属性有课程号cid(int), 课程名cname(char), 前导课pre(char), 地点caddress(char), 开始周startw(int), 结束周endw(int), 课程时间ctime(char), 编写sql语句按照结束周降序显示课程号等于11的课程"
# temptext7="在表格用户(user)中,包含的属性有用户号uid(int), 用户名username(char), 姓名name(char), 密码pass(char), 权限provilige(char), 发帖数postnum(int), 好评数goodnum(int), 好友数friendnum(int), 差评数poornum(int), 编写sql语句显示好评数等于9的用户的用户名、密码、发帖数信息"
# temptext8="在表格课程信息(courseinfo)中,包含的属性有课程号cid(int), 课程名cname(char), 前导课pre(char), 地点caddress(char), 开始周startw(int), 结束周endw(int), 课程时间ctime(char), 编写sql语句按照开始周降序显示地点为'R'的课程信息"
# sqltext5="select all from students where (age=6 OR soft>8)"
# sqltext4="select all from students where (class='F' AND prog<14)"
# sqltext1="select all from students where sid=8 orderby sid dec"
# sqltext2="select all from item where (sprice>=10 OR provider='F') orderby num dec"
# sqltext3="select all from students where (prog>=5 OR sid=10)"
# sqltext6="select all from course_info where cid=11 orderby endw dec"
# sqltext7="select (username,pass,postnum) from user where goodnum=9"
# sqltext8="select all from courseinfo where caddress='R'"

algOpTextEn=['=','<>','>=','<=','>','<','like']
algOpTextZh=['等于','不等于','大于等于','小于等于','大于','小于','类似于']
cateOpEn=['=']
cateOpZh=['为']

sample={}

def find_all(sub, s):
    index_list = []
    index = s.find(sub)
    while index != -1:
        index_list.append(index)
        index = s.find(sub, index + 1)

    if len(index_list) > 0:
        return index_list
    else:
        return [-1]

def textReplace(text, s):
    newtext=text
    if (s>0):  #requirement preprocessing sql
        posv = find_all(sample['tablezh'], newtext)
        pos1= posv[-1]
        if (pos1>=0):
            newtext=newtext[:pos1]+newtext[pos1:].replace(sample['tablezh'],'table')
        latt=len(sample['attribute'])
        for i in range(latt):
            pos1 = newtext.find(sample['attribute'][i]['zh'])
            if (pos1 >= 0):
                newtext = newtext[:pos1]+newtext[pos1:].replace(sample['attribute'][i]['zh'], 'att'+str(i))
        #calculate OP processing
        algOpNum=len(algOpTextZh)
        for i in range(algOpNum):
            posv=find_all(algOpTextZh[i], newtext)
            if (len(posv)==1 and posv[0]<0): continue
            else:
                offset=0
                li=len(algOpTextZh[i])
                for j in range(len(posv)):
                    pos1=posv[j]+offset
                    if (pos1>0 and newtext[pos1-1:pos1].isdigit() and newtext[pos1+li:pos1+li+1].isdigit()):
                        pos1+=li
                        for j in range(5):
                            if (newtext[pos1+j].isdigit()): continue
                            else: break
                        digitValue=newtext[pos1:pos1+j]
                        pos2=newtext[:pos1+j].rfind("att")
                        if (pos2>=0):
                            index=newtext[pos2+3:pos1-len(algOpTextZh[i])]
                            newtext=newtext[:pos1]+newtext[pos1:pos1+j+1].replace(digitValue, "value"+index)+newtext[pos1+j+1:]
                            offset+=len("value"+index)-len(digitValue)
                            sample['attribute'][int(index)]['value']=str(digitValue)

        #category OP processing
        cateOpNum=len(cateOpZh)
        for i in range(cateOpNum):
            pos1=newtext.find(cateOpZh[i])
            cateValue=""
            if (pos1>=0):
                newtext1=newtext[:pos1]
                newtext2=newtext[pos1:]
                # pos1+=len(cateOpZh[i])
                pos2=newtext2.find("'")
                for j in range(1,5):
                    if (newtext2[pos2+1+j]=="'"): break
                cateValue=newtext2[pos2+1:pos2+1+j]
                pos4 = newtext1.rfind("att")
                if (pos4>=0):
                    index=newtext1[pos4+3:]
                    newtext=newtext1+newtext2[pos2-len(cateOpZh[i]):].replace(cateValue, "value"+index)
                    sample['attribute'][int(index)]['value']="'"+cateValue+"'"

    else:   #sql statement preprocessing
        pos1=newtext.find(sample['tableen'])
        if (pos1>=0):
            newtext=newtext.replace(sample['tableen'],'table')
        latt=len(sample['attribute'])
        for i in range(latt):
            pos1 = newtext.find(sample['attribute'][i]['en'])
            if (pos1 >= 0):
                newtext = newtext.replace(sample['attribute'][i]['en'], 'att'+str(i))

        #calculate OP processing
        algOpNum=len(algOpTextEn)
        for i in range(algOpNum):
            posv=find_all(algOpTextEn[i], newtext)
            if (len(posv)==1 and posv[0]<0): continue
            else:
                li = len(algOpTextEn[i])
                for j in range(len(posv)):
                    pos1=posv[j]
                    if (li == 1 and (newtext[pos1+1:pos1+2] in ['>', '<', '='])): continue
                    if (li==1 and (newtext[pos1-1:pos1] in ['>','<','='])): continue
                    if (pos1>0 and newtext[pos1-1:pos1].isdigit() and (newtext[pos1+li:pos1+li+1].isdigit() or newtext[pos1+li:pos1+li+1]=="'")):
                        pos1+=li
                        # pos2=newtext[:pos1].rfind("att")
                        for j1 in range(5):
                            if (newtext[pos1+j1:pos1+j1+1].isdigit()): continue
                            else: break
                        pos2=newtext[:pos1+j1].rfind("att")

                        if (pos2>=0):
                            index=newtext[pos2+3:pos1-len(algOpTextEn[i])]
                            if (sample['attribute'][int(index)]['type']=='int'):
                                digitValue = newtext[pos1:pos1 + j1]
                                if sample['attribute'][int(index)].get('value'):
                                    digitValue= str(sample['attribute'][int(index)]['value'])
                                else:
                                    sample['attribute'][int(index)]['value']=digitValue
                                newtext=newtext[:pos1].replace('att'+str(index), 'att'+str(index)+' ')+newtext[pos1:pos1+len(digitValue)+1].replace(digitValue, " value"+index)+newtext[pos1+len(digitValue)+1:]
                            else:
                                cateValue = sample['attribute'][int(index)]['value']
                                # newtext = newtext[:pos1].replace('att'+str(index), 'att'+str(index)+' ')+newtext[pos1:pos1+len(cateValue)+1].replace(cateValue," 'value" + index + "'")+newtext[pos1+len(cateValue)+1:]
                                newtext = newtext[:pos1].replace('att'+str(index), 'att'+str(index)+' ')+newtext[pos1:pos1+len(cateValue)+1].replace(cateValue," value" + index)+newtext[pos1+len(cateValue)+1:]

    return newtext



# def oldstandarizeRequirement(text,sqltext):
#     newtext="在表格"
#     newsql="select"
#
#     textlist = text.split(',')
#     text = textlist[0].strip()
#     pos1 = text.find('表格')
#     pos2 = text.find('(')
#     pos3 = text.find(')')
#     sample['tablezh'] = text[pos1+2:pos2]
#     sample['tableen'] = text[pos2 + 1: pos3]
#     newtexttable='table'+text[pos3+1:]+","
#     attnum = len(textlist)
#     att = []
#     newtextatt=""
#     for i in range(1,attnum - 1):   #previous attnum -2
#         attelem = {}
#         text = textlist[i].strip()
#         if(i==1):
#             pos1=text.find('属性有')
#             text=text[pos1+3:]
#             #newtextatt+= text[:pos1+3]
#         pos2=0
#         for j in range(len(text)):
#             if text[j].isascii()==True:
#                 break
#             else:
#                 pos2+=1
#         attelem['zh']=text[:pos2]
#         temptext = text[pos2:]
#         pos3 = temptext.find('(')
#         attelem['en']=temptext[:pos3]
#         attelem['type'] = temptext[pos3 + 1:-1]
#         attelem['order'] = i
#         att.append(attelem)
#         newtextatt+="att"+str(i)+","
#     sample['attribute'] = att
#
#     text = textlist[attnum-1].strip()
#     if text.find('sql')>0:
#         newlasttext =textReplace(text, 1)
#         newtext=newtext+newtexttable+newtextatt+newlasttext
#         sqltext =textReplace(sqltext,0)
#     else:
#         newtext='输入文本不符合规范'
#     return newtext, sqltext



def standarizeRequirement(text,sqltext):
    newtext="在表格"
    newsql="select"

    textlist = text.split(',')
    text = textlist[0].strip()
    pos1 = text.find('表格')
    pos2 = text.find('(')
    pos3 = text.find(')')
    sample['tablezh'] = text[pos1+2:pos2]
    sample['tableen'] = text[pos2 + 1: pos3]
    newtexttable='table'+text[pos3+1:]+","
    attnum = len(textlist)
    att = []
    newtextatt=""
    for i in range(1,attnum - 1):   #previous attnum -2
        attelem = {}
        text = textlist[i].strip()
        if(i==1):
            pos1=text.find('属性有')
            text=text[pos1+3:]
            #newtextatt+= text[:pos1+3]
        pos2=0
        for j in range(len(text)):
            if text[j].isascii()==True:
                break
            else:
                pos2+=1
        attelem['zh']=text[:pos2]
        temptext = text[pos2:]
        pos3 = temptext.find('(')
        attelem['en']=temptext[:pos3]
        attelem['type'] = temptext[pos3 + 1:-1]
        attelem['order'] = i
        att.append(attelem)
        newtextatt+="att"+str(i-1)+","
    sample['attribute'] = att

    text = textlist[attnum-1].strip()
    if text.find('sql')>0:
        newlasttext =textReplace(text, 1)
        newtext=newtext+newtexttable+newtextatt+newlasttext
        sqltext =textReplace(sqltext,0)
    else:
        newtext='输入文本不符合规范'
    return newtext, sqltext

sample={}
# a,b=standarizeRequirement(temptext,"")
# text,sqltext =standarizeRequirement(temptext1,sqltext1)





# zhtext_file="D:/data/SQLRequirement.txt"
# sqltext_file="D:/data/SQLStatement.txt"
#
# with open(zhtext_file,'r',encoding='utf-8') as zhf:
#     zhlines=zhf.read().split("\n")[:-1]
#
# with open(sqltext_file,'r',encoding='utf-8') as sqlf:
#     sqllines=sqlf.read().split("\n")[:-1]
#
#
# line_num=len(zhlines)
# while line_no<line_num:
#     line=zhlines[line_no]
#     sqlline=sqllines[line_no]
#     sample = {}
#     print(line_no)
#     # print(line)
#     # print(sqlline)
#     if 0<len(line)<=sequence_length:
#         fulltext, sqltext = standarizeRequirement(line, sqlline)
#         splits = jieba.cut(fulltext.strip(), cut_all=False)
#         #splits = [term.encode("utf8", "ignore") for term in splits]
#         text=""
#         for split in splits:
#             text += split+" "
#         chinese=text.rstrip()
#
#         sqlstatement="[startstart] "+sqltext+" [endend]"
#         sqlsplits = jieba.cut(sqlstatement.strip(), cut_all=False)
#         sqlstr=""
#         for sqlsplit in sqlsplits:
#             sqlstr += sqlsplit+" "
#         sql = sqlstr.rstrip()
#
#         text_pairs.append((chinese,sql))
#     line_no+=1


import random
# print(random.choice(text_pairs))



def generateValidData(zhvaltext_file, sqlvaltext_file):
    with open(zhvaltext_file, 'r', encoding='utf-8') as zhvalf:
        zhvallines = zhvalf.read().split("\n")[:-1]

    with open(sqlvaltext_file, 'r', encoding='utf-8') as sqlvalf:
        sqlvallines = sqlvalf.read().split("\n")[:-1]

    valtext_pairs = []
    valline_no = 0
    validnum=0
    valline_num = len(zhvallines)
    while valline_no < valline_num:
        valline = zhvallines[valline_no]
        sqlvalline = sqlvallines[valline_no]
        valsample = {}
        # print(valline_no)
        # print(line)
        # print(sqlline)
        if 0 < len(valline) <= sequence_length:
            validFlag=True
            fullvaltext, sqlvaltext = standarizeRequirement(valline, sqlvalline)
            valsplits = jieba.cut(fullvaltext.strip(), cut_all=False)
            # splits = [term.encode("utf8", "ignore") for term in splits]
            valtext = ""
            for valsplit in valsplits:
                valtext += valsplit + " "
            chineseval = valtext.rstrip()
            checklist=["'value5'", "'r'", "'q'", "'value7'", "'h'", "'n'", "'b'", "'g'", "'d'", "'t'", "'f'", "'l'", "'c'", "'o'", "'p'", "'s'", "'e'", "'j'", "15", "1", "'k'", "'m'", "14", "2", "'i'", "4", "12", "8", "5", "11", "7", "18", "3", "9", "16", "17", "19", "10", "6", "13", "'value6'", "'value8'", "value76", "value54", "value74", "value52", "value00", "'value9'", "value97", "value85", "value79", "value78", "value72", "value68", "value67", "value66", "value65", "value60", "value57", "value48", "value45", "value42", "value07", "value04", "value03"]
            sqlvalstatement = "[startstart] " + sqlvaltext + " [endend]"
            # sqlvalsplits = jieba.cut(sqlvalstatement.strip(), cut_all=False)
            # sqlvalsplits=sqlvalstatement.split(" ()>=<,")
            sqlvalsplits = re.split("[ (),]", sqlvalstatement)
            sqlvalstr = ""
            for sqlvalsplit in sqlvalsplits:
                if sqlvalsplit in checklist:
                    print(sqlvalstatement)
                    print(sqlvalline)
                    validFlag=False
                sqlvalstr += sqlvalsplit + " "
            sqlval = sqlvalstr.rstrip()
            if validFlag==True:
                valtext_pairs.append((chineseval, sqlval))
                validnum+=1
            valline_no += 1
    return valtext_pairs, validnum  #valline_no


zhtext_file="D:/data/SQLRequirement.txt"
sqltext_file="D:/data/SQLStatement.txt"
text_pairs, line_no=generateValidData(zhtext_file, sqltext_file)


valtext_pairs=[]

zhvaltext_file = "D:/data/valSQLRequirement.txt"
sqlvaltext_file = "D:/data/valSQLStatement.txt"
valtext_pairs, valline_no=generateValidData(zhvaltext_file, sqlvaltext_file)


train_pairs=text_pairs
random.shuffle(valtext_pairs)
num_val_samples=int(0.5*len(valtext_pairs))
# num_test_samples=len(valtext_pairs)-num_val_samples

val_pairs=valtext_pairs[:num_val_samples]
test_pairs=valtext_pairs[num_val_samples+1:]


import tensorflow as tf
import string

# strip_chars=string.punctuation

strip_chars=" "
strip_chars=strip_chars.replace('[','') # 把[换成空，说明不去掉它
strip_chars=strip_chars.replace(']','')

def custom_standardization(input_string):  # 自定义标准化函数
    # input_string = input_string.replace('[','')
    # input_string = input_string.replace(']', '')
    lowercase=tf.strings.lower(input_string)  # 先转成小写
    # return lowercase
    return tf.strings.regex_replace(lowercase,'[\[\]]',"")  # 去掉[和])
    # return tf.strings.regex_replace(lowercase,f'[{re.escape(strip_chars)}]')  # 保留[和]，去掉¿


source_vectorization=TextVectorization(max_tokens=vocab_size,output_mode='int',standardize=custom_standardization, output_sequence_length=sequence_length)
# 源语言（英语）的词嵌入

target_vectorization=TextVectorization(max_tokens=vocab_size,output_mode='int',standardize=custom_standardization, output_sequence_length=sequence_length+1)
# 目标语言（中文）的词嵌入，生成的中文语句子多了一个词元，因为在训练的时候需要将句子偏移一个时间步

train_chinese_texts=[pair[0] for pair in train_pairs]
train_sql_texts=[pair[1] for pair in train_pairs]

source_vectorization.adapt(train_chinese_texts)
target_vectorization.adapt(train_sql_texts)
# 学习词表，给每个单词一个编号

# 将词汇表保存
source_vocab=source_vectorization.get_vocabulary()
target_vocab=target_vectorization.get_vocabulary()

source_vocab_file= "D:/data/source_vocab.json"
target_vocab_file= "D:/data/target_vocab.json"
#
import json
with open(source_vocab_file, 'w', encoding='utf-8') as fout:
    json.dump(source_vocab, fout, ensure_ascii=False)


with open(target_vocab_file, 'w', encoding='utf-8') as fout:
    json.dump(target_vocab, fout, ensure_ascii=False)
#
# json_file = open(source_vocab_file, 'r', encoding='utf-8')
# source_vocab2 = json.load(json_file)
# json_file.close()
#
# json_file = open(target_vocab_file, 'r', encoding='utf-8')
# target_vocab2 = json.load(json_file)
# json_file.close()



batch_size=32
def format_dataset(chinese,sql):
    eng=source_vectorization(chinese)
    spa=target_vectorization(sql)
    return ({
        'chinese':eng,
        'sql':spa[:,:-1] # 输入中文句子不包含最后一个词元，保证输入和目标有相同长度
    },spa[:,1:]) # 目标中文句子向后偏移一个时间步，二者长度相同，都是20个单词

def make_dataset(pairs): # 中文->sql的pair列表
    ch_texts,sql_texts=zip(*pairs) # 把英语和中文分别放在两个数组
    ch_texts=list(ch_texts)
    sql_texts=list(sql_texts)
    dataset=tf.data.Dataset.from_tensor_slices((ch_texts,sql_texts)) # 生成中文和sql的tf.data管道
    dataset=dataset.batch(batch_size) # 设置批尺寸dataset = {ParallelMapDataset: 1302} <ParallelMapDataset element_spec=({'english': TensorSpec(shape=(None, 20), dtype=tf.int64, name=None), 'spanish': TensorSpec(shape=(None, 20), dtype=tf.int64, name=None)}, TensorSpec(shape=(None, 20), dtype=tf.int64, name=None))>
    dataset=dataset.map(format_dataset,num_parallel_calls=10) # 对每一个批应用该函数
    return dataset.shuffle(2048).prefetch(16).cache() # 利用内存缓存来加快预处理速度

train_ds=make_dataset(train_pairs)
val_ds=make_dataset(val_pairs)

for inputs,targets in train_ds.take(1):# 取一个批
    print(f'inputs["chinese"].shape: {inputs["chinese"].shape}')
    print(f'inputs["sql"].shape: {inputs["sql"].shape}')
    print(f'targets.shape:{targets.shape}')

# keras.backend.clear_session()

class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, mask=None):
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, :], dtype="int32")
        attention_output = self.attention(
            query=inputs, value=inputs, key=inputs, attention_mask=padding_mask
        )
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)

    def get_config(self):
        config=super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads":self.num_heads,
            "dense_dim":self.dense_dim,
        })
        return config


class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=embed_dim
        )
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)
    def get_config(self):
        config=super().get_config()
        config.update({
            "embed_dim":self.embed_dim,
            "sequence_length": self.sequence_length,
            "vocab_size":self.vocab_size,
        })
        return config


class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, latent_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.attention_2 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.dense_proj = keras.Sequential(
            [layers.Dense(latent_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()
        self.supports_masking = True

    def get_config(self):
        config=super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "latent_dim": self.latent_dim,
        })
        return config

    def call(self, inputs, encoder_outputs, mask=None):
        causal_mask = self.get_causal_attention_mask(inputs)
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, :], dtype="int32")
            padding_mask = tf.minimum(padding_mask, causal_mask)

        attention_output_1 = self.attention_1(
            query=inputs, value=inputs, key=inputs, attention_mask=causal_mask
        )
        out_1 = self.layernorm_1(inputs + attention_output_1)

        attention_output_2 = self.attention_2(
            query=out_1,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=padding_mask,
        )
        out_2 = self.layernorm_2(out_1 + attention_output_2)

        proj_output = self.dense_proj(out_2)
        return self.layernorm_3(out_2 + proj_output)

    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)],
            axis=0,
        )
        return tf.tile(mask, mult)

embed_dim = 128   #256
latent_dim = 1024
num_heads = 12

encoder_inputs = keras.Input(shape=(None,), dtype="int64", name="chinese")
x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(encoder_inputs)
encoder_outputs = TransformerEncoder(embed_dim, latent_dim, num_heads)(x)
#encoder = keras.Model(encoder_inputs, encoder_outputs)

decoder_inputs = keras.Input(shape=(None,), dtype="int64", name="sql")
#encoded_seq_inputs = keras.Input(shape=(None, embed_dim), name="decoder_state_inputs")
x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(decoder_inputs)
x = TransformerDecoder(embed_dim, latent_dim, num_heads)(x, encoder_outputs)
x = layers.Dropout(0.5)(x)
decoder_outputs = layers.Dense(vocab_size, activation="softmax")(x)
#decoder = keras.Model([decoder_inputs, encoded_seq_inputs], decoder_outputs)

#decoder_outputs = decoder([decoder_inputs, encoder_outputs])
transformer = keras.Model(
    [encoder_inputs, decoder_inputs], decoder_outputs)



callbacks_list=[
    keras.callbacks.ModelCheckpoint(
        filepath="D:/temp/QA/TransformerSQL",
        monitor="val_loss",
        save_best_only=True)
]


epochs = 20  # This should be at least 30 for convergence


transformer.compile(
    optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)
transformer.summary()


# model training
transformer.fit(train_ds, epochs=epochs, validation_data=val_ds,callbacks=callbacks_list)

transformer.save_weights("TransformerSQLModelWeights7.h5")


# from keras.models import load_model
# from keras_bert import get_custom_objects

# transformer=keras.models.load_model("TransformerSQLModelWeights.h5")

#weight3_ver   vocab_size=2100 settings embed_dim = 256  latent_dim = 2048 num_heads = 8  epochs = 30  datav1
#weight4_ver   vocab_size=1000 settings embed_dim = 256  latent_dim = 1024 num_heads = 12  epochs = 20  datav2
#weight5_ver   vocab_size=800 settings embed_dim = 128  latent_dim = 1024 num_heads = 12  epochs = 30 datav3
#weight6_ver   vocab_size=800 settings embed_dim = 128  latent_dim = 1024 num_heads = 12  epochs = 40 datav4
#weight7_ver   vocab_size=800 settings embed_dim = 128  latent_dim = 1024 num_heads = 12  epochs = 20 datav4  修改了sqlstatements的分词方法



# transformer.load_weights('TransformerSQLModelWeights7.h5')


import numpy as np
spa_vocab = target_vectorization.get_vocabulary()
spa_index_lookup = dict(zip(range(len(spa_vocab)), spa_vocab))
max_decoded_sentence_length = 60  #20

# transformer=keras.models.load_model("D:/temp/QA/TransformerSPATrans")

def decode_sequence(input_sentence):
    tokenized_input_sentence = source_vectorization([input_sentence])
    decoded_sentence = "[start]"
    for i in range(max_decoded_sentence_length):
        tokenized_target_sentence = target_vectorization([decoded_sentence])[:, :-1]
        #predictions = transformer([tokenized_input_sentence, tokenized_target_sentence])
        next_token_predictions = transformer.predict([tokenized_input_sentence, tokenized_target_sentence])

        #sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token_index = np.argmax(next_token_predictions[0, i, :])
        sampled_token = spa_index_lookup[sampled_token_index]
        decoded_sentence += " " + sampled_token

        # if sampled_token == "[end]":
        if sampled_token.find("endend")>=0:
            break
    return decoded_sentence


def deStandarize(text):
    tablename=sample['tableen']
    text1 = text.replace("from table ", "from "+tablename+" ")
    text=text1
    posv=find_all("att",text)
    lenatt=len("att")
    lp=len(posv)
    postfix = []
    for i in range(lp): postfix.append(text[posv[i]+lenatt:posv[i] + lenatt+1])
    postfix = set(postfix)
    postfix=list(postfix)
    offset=0
    for i in range(lp):
        pos1 = posv[i]+offset
        if(pos1<0): break
        for j in range(1,5):
            if (pos1 >=0 and text[pos1+lenatt+j:pos1+lenatt+j].isdigit()): continue
            else: break
        attindex=text[pos1+lenatt:pos1+lenatt+j].strip()
        if (postfix.count(attindex)<=0):continue
        atttemp="att"+attindex
        attname=sample['attribute'][int(attindex)]['en']
        text = text.replace(" "+atttemp+" ", " "+attname+" ")
        offset+=len(attname)-len(atttemp)

    posv=find_all("value",text)
    lenvalue=len("value")
    lp=len(posv)
    offset=0
    for i in range(lp):
        pos1 = posv[i]+offset
        if(pos1<0): break
        for j in range(1,5):
            if (pos1 >=0 and text[pos1+lenvalue+j:pos1+lenvalue+j].isdigit()): continue
            else: break
        valueindex=text[pos1+lenvalue:pos1+lenvalue+j].strip()
        valuetemp="value"+valueindex
        if ('value' in sample['attribute'][int(valueindex)].keys()):
            valuereal=sample['attribute'][int(valueindex)]['value']
        else:
            valuereal="wrongindex"
        text = text.replace(valuetemp, valuereal)
        offset+=len(valuereal)-len(valuetemp)

    attl=len(sample['attribute'])
    for i in range(attl):
        attin='att'+str(i)
        atten=sample['attribute'][i]['en']
        text=text.replace(attin+" ", atten+" ")
        text = text.replace(" "+attin, " "+ atten)

    text=text.replace("' '","'")
    return text


def testtext(text):
    temptext=text
    fulltext,sqltext =standarizeRequirement(temptext,"")
    print(fulltext)
    splits = jieba.cut(fulltext.strip(), cut_all=False)
    # splits = [term.encode("utf8", "ignore") for term in splits]
    text = ""
    for split in splits:
        text += split + " "
    chinese = text.rstrip()
    translated = decode_sequence(chinese)
    print(translated)
    # translated =translated.replace("startstart","")
    translated = translated.replace("[start]", "")
    # translated = translated.replace("select all ", "select * ")
    pos=translated.find(" endend")
    if (pos>=0):
        translatedtemp=translated[:pos]
    else:
        translatedtemp="wrong output"

    decode_text=deStandarize(translatedtemp)
    decode_text = decode_text.replace("select all ", "select * ")
    return decode_text

sample={}
temptext=[]
temptext.append("在表格文档列表(doc)中,包含的属性有文档号did(int), 文档名dname(char), 类别class(char), 单词数wordnum(int), 文章数papernum(int), 请写出sql语句按照文档号降序列出单词数大于100或者文章数大于10的文档列表的文档名信息")
temptext.append("在表格职员信息(employee)中,包含的属性有工号eid(int), 姓名name(char), 部门depart(char), 年龄age(int), 性别sex(char), 工龄workyear(int), 基本工资bsalary(int), 奖金psalary(int), 补贴butie(int), 请写出sql语句按照工号降序列出基本工资大于2000的职员信息的工号、姓名、奖金、基本工资信息")
temptext.append("在表格学生记录(students)中,包含的属性有学号sid(int), 姓名sname(char), 班级class(char), 年龄age(int), 性别sex(char), 程序设计prog(int), 软件工程soft(int), 人工智能inte(int), 数据挖掘data(int), 编写sql语句按照学号升序显示班级为'01'并且程序设计大于70的学生记录的姓名、班级信息")
temptext.append("在表格文档列表(doc)中,包含的属性有文档号did(int), 文档名dname(char), 类别class(char), 单词数wordnum(int), 文章数papernum(int), 请写出sql语句列出单词数大于100的文档列表")
temptext.append("在表格职员信息(employee)中,包含的属性有工号eid(int), 姓名name(char), 部门depart(char), 年龄age(int), 性别sex(char), 工龄workyear(int), 基本工资bsalary(int), 奖金psalary(int), 补贴butie(int), 请写出sql语句列出基本工资小于2000的职员信息")
temptext.append("在表格学生记录(students)中,包含的属性有学号sid(int), 姓名sname(char), 班级class(char), 年龄age(int), 性别sex(char), 程序设计prog(int), 软件工程soft(int), 人工智能inte(int), 数据挖掘data(int), 编写sql语句显示数据挖掘不等于60的学生记录")



lt=len(temptext)
for i in range(lt):
    text=testtext(temptext[i])
    print("要求: "+temptext[i])
    print("sql: "+text)

# keras.backend.clear_session()
# test_chinese_texts = [pair[0] for pair in test_pairs]
# ltest=len(test_chinese_texts)
# for i in range(5):
#     rint=random.randint(0,ltest-1)
#     input_sentence = test_chinese_texts[rint]
#     translated = decode_sequence(input_sentence)
#     translated =translated.replace("[start]","")
#     translated =translated.replace("end", "")
#     print(f"{rint}---\n")
#     print(input_sentence)
#     print(translated)

