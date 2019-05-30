# sentiment-analysis
> 算分project 备份
>
> kaggle比赛：<https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews>

- 实现模型：
  - Lstm
  - TextCNN
  - Recursive
  - TreeLstm
- 数据集：Sentiment Treebank

- 提供预训练词向量接口：GloVe

- 展示网站：django
- 另提供wechatapi



- bug：

  - 模块间库的导入方式存在相对路径的问题，具体为：

    ```python
    from . import xxx
    # 貌似import xxx是不规范的，能力有限，有时间好好学习一下
    ```

  - 没有认真炼丹调参，模型可能未处于最优状态

  - TreeLstm如果希望重视Root_acc，感觉可以修改Root的loss权重