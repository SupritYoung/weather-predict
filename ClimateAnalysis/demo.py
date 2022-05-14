from collections import Counter
import jieba
import torchtext

# 实现中文分词
def tokenizer(corpus):
    tokenized_corpus = jieba.cut(corpus) # 调用 jieba 分词
    tokenized_corpus = [line for line in tokenized_corpus] # 去掉回车符，转为list类型
    return tokenized_corpus

# 使用 torchtext.vocab 快速构建词表
def build_vocab(tokenizer, corpus, min_freq, specials=None):
    if specials is None:
        specials = ['<unk>', '<pad>', '<bos>', '<eos>']
    counter = Counter()
    for line in corpus:
        counter.update(tokenizer(line))
    return vocab(counter, min_freq=min_freq, specials=specials)

vocab = build_vocab(tokenizer, corpus, min_freq=1, specials=['<unk>', '<pad>', '<bos>', '<eos>'])
