import nltk
from nltk.tokenize import word_tokenize

# =========================================================
# 1. 核心修复：路径注入
# =========================================================
nltk.data.path.append("/root/nltk_data/")

# 全局缓存，防止重复加载
_wordnet_cache = None

def get_wordnet():
    global _wordnet_cache
    if _wordnet_cache is None:
        try:
            from nltk.corpus import wordnet
            wordnet.synsets('drink') # 触发加载
            _wordnet_cache = wordnet
        except Exception:
            return None
    return _wordnet_cache

# =========================================================
# 2. WUP 计算逻辑
# =========================================================
def wup(w1, w2, alpha):
    wn = get_wordnet()
    if wn is None: return 0.0

    try:
        # 过滤非字母数字字符
        if not w1.isalnum() or not w2.isalnum():
            return 0.0
            
        ss1 = wn.synsets(w1)
        ss2 = wn.synsets(w2)
        if not ss1 or not ss2: return 0.0
            
        max_sim = 0.0
        for s1 in ss1:
            for s2 in ss2:
                try:
                    sim = s1.wup_similarity(s2)
                except:
                    sim = 0.0
                if sim is not None and sim > max_sim:
                    max_sim = sim
        
        if max_sim < alpha:
            return 0.1 * max_sim
        return max_sim
    except:
        return 0.0

def wups(words1, words2, alpha):
    sim = 1.0
    flag = False
    for w1 in words1:
        max_sim = 0
        for w2 in words2:
            word_sim = wup(w1, w2, alpha)
            if word_sim > max_sim:
                max_sim = word_sim
        
        # 防止乘法归零
        if max_sim == 0: 
            max_sim = 0.000001 
            
        sim *= max_sim
        flag = True
    if not flag:
        sim = 0.0
    return sim

def get_wups(pred, truth, alpha=0.9):
    # =================================================
    # 3. 核心修复：智能处理 List 和 String
    # =================================================
    
    # 处理 Pred
    if isinstance(pred, list):
        pred_tokens = pred
    else:
        pred_str = str(pred)
        # 尝试解析字符串形式的列表
        if pred_str.strip().startswith('[') and pred_str.strip().endswith(']'):
            try:
                pred_tokens = eval(pred_str)
            except:
                pred_tokens = word_tokenize(pred_str)
        else:
            pred_tokens = word_tokenize(pred_str)

    # 处理 Truth
    if isinstance(truth, list):
        truth_tokens = truth
    else:
        truth_str = str(truth)
        if truth_str.strip().startswith('[') and truth_str.strip().endswith(']'):
            try:
                truth_tokens = eval(truth_str)
            except:
                truth_tokens = word_tokenize(truth_str)
        else:
            truth_tokens = word_tokenize(truth_str)

    # 计算分数
    item1 = wups(pred_tokens, truth_tokens, alpha)
    item2 = wups(truth_tokens, pred_tokens, alpha)
    value = min(item1, item2)

    return value
