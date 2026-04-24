# 一、本次作業的核心目標
這次作業是 Artificial Intelligence NYCU Spring 2026 Project #2，截止日是 2026/5/3。核心目標是：實作並評估一個用於 representation learning 的 foundation model，方法採用 self-supervised learning, SSL，具體以 SimCLR / contrastive learning 為主。作業指定先從較簡單的設定開始：使用 modified ResNet-18 作為 backbone，資料集使用 CIFAR-10 32×32 小影像，在 baseline 成功後再進行各種變化與分析。

簡單說，你要做的不是只訓練一個 CIFAR-10 分類器，而是要先用 無標籤式的對比學習 訓練出一個 encoder/backbone，使它能產生好的 image representation；接著再用 linear probing 檢驗這個 representation 好不好。

這個設定直接對應 SimCLR 論文的精神：SimCLR 透過對同一張影像產生兩個不同 augmentation view，讓模型學會最大化同源 view 的一致性；論文也指出有效的資料增強、nonlinear projection head、normalized temperature-scaled cross entropy loss，以及較大的 batch size 對 contrastive learning 很重要。
