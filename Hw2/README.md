# SimCLR on CIFAR-10: Representation Learning Analysis
## Artificial Intelligence NYCU Spring 2026 Project #2
# Abstract
本作業旨在探討自監督學習（Self-Supervised Learning, SSL）在無標註資料情境下的影像表示學習能力，並以 SimCLR 對比式學習架構應用於 CIFAR-10 資料集進行實驗分析。本次實作以 Jupyter Notebook 作為主要開發與實驗環境，完成資料載入、模型建構、訓練流程、評估方法、結果視覺化與實驗分析。SimCLR 透過對同一張影像產生兩個不同的資料增強版本，建立正樣本對（positive pairs），並利用 NT-Xent 損失函數，使模型在不使用類別標籤的情況下學習具有判別力的影像特徵表示。

在主要實驗中，本作業使用經過修改的 ResNet-18 作為 backbone encoder，將第一層卷積改為適合 CIFAR-10 影像大小的 3×3 convolution，並移除初始 max-pooling，以避免過早下採樣。SimCLR 訓練時在 encoder 後接上 projection head 進行對比學習；訓練完成後移除 projection head，凍結 encoder，並透過 linear probing 評估 learned representation 的品質。此外，本作業亦建立兩個比較基準：其一為使用標註資料從頭訓練的 supervised learning baseline，其二為隨機初始化且凍結的 backbone 搭配 linear classifier，作為 lower-bound baseline。
實驗結果顯示，SimCLR 模型在 CIFAR-10 linear probing 下達到 84.4% 測試準確率，明顯優於 random frozen backbone baseline 的 34.4%，表示 SimCLR 即使不使用標籤，也能學習到具有分類能力的影像表示。然而，supervised learning baseline 達到 92.97% 測試準確率，顯示在有標註資料的情境下，直接最佳化分類目標仍具有較高效能。此外，訓練過程中的 kNN monitor accuracy 與最終 linear probing accuracy 接近，顯示 kNN monitor 可作為觀察 representation quality 的有效指標。

除了主要實驗外，本作業亦規劃並加入四項延伸實驗以進一步分析 SimCLR 的關鍵因素。Experiment 1 探討 NT-Xent loss 中 temperature 參數對 loss curve 與 kNN monitor accuracy 的影響。Experiment 2 分析 batch size 對 contrastive learning 的影響，特別是 negative samples 數量改變時 representation learning 的差異。Experiment 3 移除 projection head，直接以 encoder output 計算 contrastive loss，以觀察 projection head 對表示品質的重要性。Experiment 4 將 CIFAR-10 上學得的 encoder 轉移至 CIFAR-100 進行 linear probing，以評估 learned representation 的跨資料集泛化能力。
綜合而言，本作業驗證了 SimCLR 對比式自監督學習在無標註影像資料上的有效性。實驗結果顯示，self-supervised learning 能大幅優於隨機特徵表示，並學習出具良好線性可分性的 visual representations；雖然其分類表現仍低於完全監督式學習，但已展現出在缺乏標註資料時作為 representation learning 方法的實用價值與潛力。

# Objective
本次作業的目標是實作並評估一個基於自監督學習（Self-Supervised Learning, SSL）的影像表示學習框架，方法採用 SimCLR，並以 CIFAR-10 資料集作為主要實驗資料。核心目的在於探討：在不使用影像類別標籤的情況下，模型是否能透過對比式學習（contrastive learning）學到具有意義且具判別能力的影像特徵表示。

在SimCLR的訓練過程中，模型會對同一張影像產生兩個不同的資料增強版本，並將它們視為正樣本對；同一個 batch 中來自不同影像的樣本則作為負樣本。透過 NT-Xent loss，模型學習使同一張影像的不同增強版本在特徵空間中更接近，而不同影像的表示則彼此分離。藉此，模型可以在無標籤資料上學習出有效的 visual representation。

完成自監督訓練後，本作業會移除 SimCLR 的 projection head，保留 backbone encoder 作為特徵萃取器，並透過 linear probing 評估其學到的表示品質。也就是凍結 encoder，只訓練一個線性分類器來進行 CIFAR-10 分類，以觀察 learned representation 是否具有良好的線性可分性。此外，為了評估 SimCLR 表示學習的效果，本作業也與兩種基準方法進行比較：
-	監督式學習模型（Supervised Learning from Scratch）：

 	使用相同的 modified ResNet-18 backbone，但直接利用 CIFAR-10 標籤從頭訓練分類模型，作為有標籤訓練下的比較基準。
-	隨機凍結 backbone 基準模型（Random Frozen Backbone Lower-Bound Baseline）：

 	使用隨機初始化且不進行訓練的backbone，僅訓練最後的線性分類器，作為 representation learning 的下界基準。

透過比較 SimCLR + linear probing、supervised learning 與 random frozen baseline 的測試準確率，本作業希望分析自監督對比式學習是否能有效學習影像特徵，並進一步理解其與監督式學習之間的效能差異。

# Methodology
本次作業採用SimCLR（[2002.05709] A Simple Framework for Contrastive Learning of Visual Representations）作為主要的自監督學習方法，並以 CIFAR-10 資料集進行影像表示學習實驗。整體方法包含三個部分：SimCLR 自監督訓練、linear probing 表示評估，以及與 supervised learning 和 random frozen baseline 的比較。
1. SimCLR Framework：
   SimCLR 是一種基於 contrastive learning 的自監督表示學習方法。其核心概念是：對同一張影像產生兩個不同的資料增強版本，並讓模型學習使這兩個版本在特徵空間中更接近；同時，來自不同影像的樣本則應在特徵空間中彼此分離。
本作業的 SimCLR 訓練流程如下：

首先，對每張 CIFAR-10 影像產生兩個獨立的 augmented views。這兩個 views 來自同一張原始影像，因此被視為 positive pair；同一個 batch 中其他影像的 augmented views 則作為 negative samples。接著，兩個 augmented views 會輸入同一個 encoder backbone，產生影像表示向量，再經過 projection head 映射到對比學習使用的 latent space。最後使用 NT-Xent loss 計算 contrastive loss，訓練模型使 positive pair 的 similarity 較高，negative pairs 的 similarity 較低。
本作業使用的主要元件如下：

![](images/SimCLR_Training.png)


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




