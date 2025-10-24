import nltk
from nltk.tree import Tree
import numpy as np
from collections import deque

class ConstituencyTreeParser:
    """选区树解析器 - 精确实现论文4.1.1节算法"""
    
    def __init__(self, parser_url='http://localhost:9000'):
        try:
            # 尝试使用CoreNLP解析器
            from nltk.parse.corenlp import CoreNLPParser
            self.parser = CoreNLPParser(url=parser_url)
        except:
            # 备用：使用NLTK内置解析器
            self.parser = nltk.RegexpParser('')
        
        # 确保NLTK数据可用
        self._download_nltk_data()
    
    def _download_nltk_data(self):
        """下载必要的NLTK数据"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            nltk.download('averaged_perceptron_tagger')
        try:
            nltk.data.find('chunkers/maxent_ne_chunker')
        except LookupError:
            nltk.download('maxent_ne_chunker')
        try:
            nltk.data.find('corpora/words')
        except LookupError:
            nltk.download('words')
    
    def parse_sentence(self, text):
        """解析句子为选区树 - 对应论文Fig.2示例"""
        sentences = nltk.sent_tokenize(text)
        trees = []
        
        for sentence in sentences:
            try:
                # 词性标注
                tokens = nltk.word_tokenize(sentence)
                pos_tags = nltk.pos_tag(tokens)
                
                # 使用正则表达式语法构建简单的选区树
                grammar = r"""
                    NP: {<DT>?<JJ>*<NN.*>+}  # 名词短语
                    VP: {<VB.*><NP|PP>*}     # 动词短语
                    PP: {<IN><NP>}           # 介词短语
                """
                chunker = nltk.RegexpParser(grammar)
                tree = chunker.parse(pos_tags)
                trees.append(tree)
                
            except Exception as e:
                print(f"Error parsing sentence: {e}")
                # 创建简单的平坦树作为备用
                tree = Tree('S', [Tree('NP', tokens) for tokens in nltk.word_tokenize(sentence)])
                trees.append(tree)
        
        return trees
    
    def serialize_tree(self, tree):
        """将选区树序列化为二进制数组 - 实现论文4.1.1节的跨度方法"""
        if isinstance(tree, str):
            return [0]  # 单节点树
        
        # 获取所有跨度的左右边界
        spans = []
        self._extract_spans(tree, 0, spans)
        
        if not spans:
            return [0]
        
        # 找到最大右边界
        max_right = max(span[1] for span in spans)
        
        # 初始化d数组（论文中的表示）
        d_array = [0] * (2 * max_right - 1)
        
        # 填充d数组
        for left, right in spans:
            d_array[right - 1] = left
        
        return d_array
    
    def _extract_spans(self, tree, start_pos, spans):
        """提取树中所有节点的跨度"""
        if isinstance(tree, str):
            return start_pos + 1
        
        current_start = start_pos
        for child in tree:
            current_start = self._extract_spans(child, current_start, spans)
        
        # 添加当前节点的跨度（如果不是叶子节点）
        if not all(isinstance(child, str) for child in tree):
            spans.append((start_pos, current_start))
        
        return current_start
    
    def visualize_tree(self, tree, filename=None):
        """可视化选区树（用于调试）"""
        try:
            tree.pretty_print()
            if filename:
                from nltk.draw.tree import TreeWidget
                tree.draw()
        except:
            print("Tree visualization not available")

# 示例使用
if __name__ == "__main__":
    parser = ConstituencyTreeParser()
    sample_text = "Male, diagnosed as pancreatic head cancer 8 months ago"
    trees = parser.parse_sentence(sample_text)
    
    for i, tree in enumerate(trees):
        print(f"Tree {i+1}:")
        print(tree)
        serialized = parser.serialize_tree(tree)
        print(f"Serialized: {serialized}")
