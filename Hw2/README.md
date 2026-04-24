# 一、本次作業的核心目標
這次作業是 Artificial Intelligence NYCU Spring 2026 Project #2，截止日是 2026/5/3。

核心目標是：實作並評估一個用於 representation learning 的 foundation model，方法採用 self-supervised learning, SSL，具體以 SimCLR / contrastive learning 為主。作業指定先從較簡單的設定開始：使用 modified ResNet-18 作為 backbone，資料集使用 CIFAR-10 32×32 小影像，在 baseline 成功後再進行各種變化與分析。

簡單說，本次要做的不是只訓練一個 CIFAR-10 分類器，而是要先用 無標籤式的對比學習 訓練出一個 encoder/backbone，使它能產生好的 image representation；接著再用 linear probing 檢驗這個 representation 好不好。

這個設定直接對應 SimCLR 論文的精神：SimCLR 透過對同一張影像產生兩個不同 augmentation view，讓模型學會最大化同源 view 的一致性；論文也指出有效的資料增強、nonlinear projection head、normalized temperature-scaled cross entropy loss，以及較大的 batch size 對 contrastive learning 很重要。

# 二、實作的 SimCLR 架構
作業聚焦在 SimCLR 第一版。SimCLR training network 包含兩個主要部分：

1. Backbone encoder：本作業使用 modified ResNet-18，去掉最後 fully connected layer。
2. Projector head：原始 SimCLR 使用 two-layer MLP，本作業建議設定為 512 -> 512 -> 128。

每個 batch 有 N 張原始圖片，要對每張圖片產生兩個獨立 augmentation，因此實際進入 contrastive loss 的樣本數是 2N。常見 augmentation 包含 random crop、horizontal flip、color jitter、random grayscale，最後還需要 normalization。

Loss 要使用 NT-Xent loss。概念是：對於 batch 中某個 augmented image x，其他 2N-1 張圖裡只有一張是它的 mate，也就是同一張原圖經另一個 augmentation 得到的 view；模型要透過 normalized projector outputs 的 cosine similarity，加上 temperature scaling 與 softmax，將 mate 找出來。該 image 的 loss 是 mate 對應 softmax 機率的 negative log，整個 batch loss 則是所有 augmented samples 的平均。

SSL 訓練結束後，projector head 要丟掉。真正拿來當 representation 的是 backbone 輸出，也就是 ResNet-18 經 global average pooling 與 flatten 之後的 512-dimensional feature。

# 三、實作上的硬性要求與建議設定
所有模型都要 from scratch 訓練，不可使用 pretrained weights。CIFAR-10 影像只有 32×32，因此標準 ResNet-18 的開頭下採樣太 aggressive，作業要求修改：

- 第一層 convolution 改成 3×3 kernel, stride=1, padding=1，取代原本 ResNet 的 7×7 stride=2。
- 第一層 convolution 後面的 max-pooling 改成 identity layer。

建議 baseline hyperparameters 如下：
| 項目                          | 建議設定                |
| --------------------------- | ------------------- |
| Optimizer                   | Adam                |
| Learning rate               | `3e-4`              |
| Weight decay                | `1e-6`              |
| Temperature                 | `0.5`               |
| Projector head              | `512 -> 512 -> 128` |
| SSL training                | 200 epochs          |
| kNN monitor                 | `k=20`              |
| kNN monitor 頻率              | 每 5 epochs 做一次      |
| Linear probing optimizer    | Adam                |
| Linear probing LR           | `1e-3`              |
| Linear probing weight decay | `1e-6`              |
| Linear probing epochs       | 100 epochs          |
| Linear probing data         | 使用完整 training set   |

Batch size 依照你的 GPU VRAM 調整：低階 GPU 可試 64，中階可試 128 或 256，較好 GPU 可試 512 或 1024。

# 四、必做實驗
作業明確說明：前兩項實驗是 required。

第一，必須完成 SSL baseline model，並報告：
- SSL training loss curve。
- kNN-monitor accuracy curve。
- 最終 linear probing result。

注意：作業特別提醒，SSL loss 雖然通常會下降，但它不一定能直接代表 representation quality。因此需要 kNN monitor 觀察 representation 是否真的變好；kNN monitor 要用 encoder output，不是 projector output。

第二，必須訓練一個 supervised learning, SL model from scratch。這個模型使用同一個 backbone，但不使用 projector，而是在 backbone 後接上 classification head，也就是 512 -> 10。你要將 SL model 的 test accuracy 與 SSL model 的 linear probing accuracy 比較，並討論兩者差異。


# 五、建議但非必做的延伸實驗
除了必做項目外，作業列出多個可探索方向。

你可以做一個真正的 lower-bound baseline：隨機初始化 backbone，然後 freeze 住不訓練，只做 linear probing。這可以和 SSL、SL 結果比較，用來確認 SSL pretraining 是否真的學到有用 representation。

Ablation study 建議優先看三類因素：
- augmentation 的種類與強度。
- temperature 的影響，例如高溫 5 或低溫 0.1 對 loss curve 與 kNN-monitor accuracy curve 的影響，並觀察兩條曲線變化是否一致。
- batch size 的影響，例如從最大可訓練 batch size 開始，逐步降到 256、128、64、32，比較 learning curves。

另外也可測：

不使用 projector head，直接用 encoder output 計算 contrastive loss。
評估時改用 projector output，也就是 128-dim representation，而不是 backbone output。
測試 representation 對新資料集的 transfer ability：freeze backbone，對另一個 dataset 做 linear probing。可選資料集包括 CIFAR-100、STL-10、Flowers-102、Food-101，或第一次作業中自己建立的資料集。這部分必須將輸入 resize 到 32×32 並 normalize。

所有 linear probing 實驗的設定必須一致，這樣不同模型或不同資料集的比較才有意義。

# 六、報告要求
最後繳交的是 PDF 報告，最多 10 頁、single-spaced。但程式碼 appendix 不算在 10 頁內。報告需要包含：
- 用 plain text 簡述你的 research question 與 motivation。
- 方法描述，包含 public libraries、open-source code、AI tools 等引用。
- 實驗描述、evaluation results、examples。
- Discussion section。
- References。
- 程式碼 appendix，從新頁開始，需組織清楚並有註解。

繳交方式是透過 E3。Late submission 最多接受 5 天，每晚一天扣 10%。檔名與第一頁都要包含 student ID。方法描述不能寫得像程式文件；讀者應該不看 code listing 也能理解你的方法。結果呈現要有足夠說明與討論，不能只丟數字；表格要用報告中正式排版的表格，不要用 screenshot；必要時使用 charts/plots。正文至少 12pt，表格與圖至少 10pt，不要為了塞進 10 頁而縮小字體或圖。

# 七、我對這份作業的精簡判斷
這份作業評分重點應該不只是 accuracy，而是你是否能完整展示：
- 你正確實作 SimCLR / NT-Xent。
- 你能用 kNN monitor 與 linear probing 評估 representation。
- 你能公平比較 SSL、SL、random baseline。
- 你能透過 ablation 解釋 augmentation、temperature、batch size、projector head 對結果的影響。
- 你能在 discussion 中提出自己的觀察，而不是只讓 AI 生成空泛說明。

最小可行完成路線是：先做 SSL baseline + kNN curve + linear probing，再做 SL from scratch comparison；若時間足夠，加上 random frozen baseline、temperature ablation、batch size ablation，報告就會比較完整。




