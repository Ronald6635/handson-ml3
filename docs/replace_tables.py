import re

# Read the HTML file
with open('機器學習與深度學習實戰速查手冊.html', 'r', encoding='utf-8') as f:
    html_content = f.read()

# Define the chapter data from the script output
chapter_data = {
    '4': '''                <tr>
                    <td><strong>線性與廣義線性模型</strong></td>
                    <td><code>LinearRegression</code></td>
                    <td><strong>線性迴歸</strong>：使用 Normal Equation 或 SVD 直接計算最佳模型參數（最小化 MSE）。</td>
                    <td>無（封閉式解法）。</td>
                    <td class="code-snippet"><code>`from sklearn.linear_model import LinearRegression; lr = LinearRegression().fit(X, y); y_pred = lr.predict(X_test)`</code><button class="copy-btn" onclick="copyCode(this)">Copy</button></td>
                </tr>  <tr>
                    <td><strong>(Linear & Generalized Linear Models)</strong></td>
                    <td><code>Ridge</code>，<code>Lasso</code>，<code>ElasticNet</code></td>
                    <td><strong>正規化線性模型</strong>：分別使用 $\ell_2$、$\ell_1$ 或兩者組合來約束權重，以減少過度擬合。</td>
                    <td><strong>正規化強度 $\alpha$</strong> (alpha)：值越大，約束越強。</td>
                    <td class="code-snippet"><code>`from sklearn.linear_model import Ridge; ridge = Ridge(alpha=0.1).fit(X, y)`</code><button class="copy-btn" onclick="copyCode(this)">Copy</button></td>
                </tr>                                                                                  <tr>
                    <td></td>
                    <td><code>LogisticRegression</code></td>
                    <td><strong>羅吉斯迴歸</strong>：用於分類任務，估計類別機率。</td>
                    <td><strong>$C$</strong> (正規化強度的倒數)；<code>penalty</code> (e.g., 'l2')。</td>
                    <td class="code-snippet"><code>`from sklearn.linear_model import LogisticRegression; log_reg = LogisticRegression(C=1.0).fit(X, y); probs = log_reg.predict_proba(X_test)`</code><button class="copy-btn" onclick="copyCode(this)">Copy</button></td>
                </tr>                                                                                                                                      <tr>
                    <td></td>
                    <td><code>SGDRegressor</code></td>
                    <td>線性模型，使用 SGD 訓練。</td>
                    <td><code>loss</code>, <code>penalty</code>, <code>alpha</code>, <code>learning_rate</code>, <code>eta0</code>。</td>
                    <td class="code-snippet"><code>`from sklearn.linear_model import SGDRegressor; sgd = SGDRegressor(loss=&#x27;squared_error&#x27;).fit(X, y)`</code><button class="copy-btn" onclick="copyCode(this)">Copy</button></td>
                </tr>                       <tr>
                    <td></td>
                    <td><code>TransformedTargetRegressor</code></td>
                    <td>對目標變數進行轉換的元估計器。</td>
                    <td><code>regressor</code>, <code>transformer</code>。</td>
                    <td class="code-snippet"><code>`from sklearn.compose import TransformedTargetRegressor; from sklearn.preprocessing import StandardScaler; ttr = TransformedTargetRegressor(regressor=LinearRegression(), transformer=StandardScaler())`</code><button class="copy-btn" onclick="copyCode(this)">Copy</button></td>
                </tr>                                                                                                                      <tr>
                    <td></td>
                    <td><code>add_dummy_feature</code></td>
                    <td>向數據集添加截距列。</td>
                    <td><code>X</code>, <code>value</code>。</td>
                    <td class="code-snippet"><code>`from sklearn.preprocessing import add_dummy_feature; X_with_dummy = add_dummy_feature(X, value=1.0)`</code><button class="copy-btn" onclick="copyCode(this)">Copy</button></td>
                </tr>                                                                                                       <tr>
                    <td></td>
                    <td><code>PolynomialFeatures</code></td>
                    <td>生成多項式和交互特徵。</td>
                    <td><code>degree</code>, <code>interaction_only</code>, <code>include_bias</code>。</td>
                    <td class="code-snippet"><code>`from sklearn.preprocessing import PolynomialFeatures; poly = PolynomialFeatures(degree=2).fit_transform(X)`</code><button class="copy-btn" onclick="copyCode(this)">Copy</button></td>
                </tr>                                                  <tr>
                    <td></td>
                    <td><code>mean_squared_error</code>, <code>root_mean_squared_error</code></td>
                    <td>均方誤差和均方根誤差。</td>
                    <td><code>y_true</code>, <code>y_pred</code>。</td>
                    <td class="code-snippet"><code>`from sklearn.metrics import mean_squared_error; mse = mean_squared_error(y_true, y_pred)`</code><button class="copy-btn" onclick="copyCode(this)">Copy</button></td>
                </tr>''',
    '5': '''                <tr>
                    <td><strong>支持向量機</strong></td>
                    <td><code>SVC</code>，<code>LinearSVC</code>，<code>SVR</code></td>
                    <td><strong>SVM 分類與迴歸</strong>：<strong>大邊界分類</strong> (Large Margin Classification)。使用 <strong>核心技巧</strong> (Kernel Trick) 處理非線性數據。</td>
                    <td><strong>$C$</strong> (正規化參數)；<strong><code>kernel</code></strong> 類型 (e.g., <strong>"rbf"</strong>, "poly")；<strong>$\gamma$</strong> (Gamma，RBF 核心參數)。</td>
                    <td class="code-snippet"><code>`from sklearn.svm import SVC; svc = SVC(kernel=&#x27;rbf&#x27;, C=1.0, gamma=&#x27;scale&#x27;).fit(X, y)`</code><button class="copy-btn" onclick="copyCode(this)">Copy</button></td>
                </tr>                                                                         <tr>
                    <td><strong>(Support Vector Machines, SVMs)</strong></td>
                    <td><code>hinge loss function</code></td>
                    <td><strong>SVM 理論基礎</strong>：硬邊界與軟邊界分類。對偶問題。</td>
                    <td>N/A；相關參數為 $C$ 和 $\gamma$。</td>
                    <td class="code-snippet"><code>N/A</code><button class="copy-btn" onclick="copyCode(this)">Copy</button></td>
                </tr>                                                                                                                 <tr>
                    <td></td>
                    <td><code>LinearSVR</code></td>
                    <td>線性支持向量迴歸。</td>
                    <td><code>epsilon</code>, <code>C</code>, <code>loss</code>。</td>
                    <td class="code-snippet"><code>`from sklearn.svm import LinearSVR; svr = LinearSVR(epsilon=0.1, C=1.0).fit(X, y)`</code><button class="copy-btn" onclick="copyCode(this)">Copy</button></td>
                </tr>                                                                                                               <tr>
                    <td></td>
                    <td><code>rbf_kernel</code></td>
                    <td>RBF 核心函數。</td>
                    <td><code>X</code>, <code>Y</code>, <code>gamma</code>。</td>
                    <td class="code-snippet"><code>`from sklearn.metrics.pairwise import rbf_kernel; K = rbf_kernel(X, Y, gamma=1.0)`</code><button class="copy-btn" onclick="copyCode(this)">Copy</button></td>
                </tr>''',
    '6': '''                <tr>
                    <td><strong>決策樹與集成方法</strong></td>
                    <td><code>DecisionTreeClassifier</code>/<code>Regressor</code></td>
                    <td><strong>決策樹</strong>：<strong>白箱模型</strong>。利用 <strong>CART 演算法</strong> 和 <strong>Gini 不純度/熵</strong> 進行分割。</td>
                    <td><strong><code>max_depth</code></strong> (最大深度)；<strong><code>min_samples_leaf</code></strong> (葉節點最小樣本數)；用於<strong>正則化</strong>。</td>
                    <td class="code-snippet"><code>`from sklearn.tree import DecisionTreeClassifier; dt = DecisionTreeClassifier(max_depth=5).fit(X, y)`</code><button class="copy-btn" onclick="copyCode(this)">Copy</button></td>
                </tr>                                                                                                                 <tr>
                    <td></td>
                    <td><code>export_graphviz</code></td>
                    <td>將決策樹導出為 GraphViz DOT 格式文件，用於可視化。</td>
                    <td><code>out_file</code>, <code>feature_names</code>, <code>class_names</code>, <code>rounded</code>, <code>filled</code>。</td>
                    <td class="code-snippet"><code>`from sklearn.tree import export_graphviz; export_graphviz(dt, out_file=&#x27;tree.dot&#x27;, feature_names=feature_names)`</code><button class="copy-btn" onclick="copyCode(this)">Copy</button></td>
                </tr>                                                                                                                    <tr>
                    <td></td>
                    <td><code>Source.from_file</code></td>
                    <td>從 DOT 文件讀取並顯示決策樹圖形，用於可視化 <code>export_graphviz</code> 導出的樹結構。</td>
                    <td><code>filepath</code> (DOT 文件路徑)；無其他參數。</td>
                    <td class="code-snippet"><code>`from graphviz import Source; graph = Source.from_file(&#x27;tree.dot&#x27;); graph.view()`</code><button class="copy-btn" onclick="copyCode(this)">Copy</button></td>
                </tr>                                 <tr>
                    <td></td>
                    <td><code>make_moons</code></td>
                    <td>生成月亮形數據集，用於測試非線性分類。</td>
                    <td><code>n_samples</code>, <code>noise</code>, <code>random_state</code>。</td>
                    <td class="code-snippet"><code>`from sklearn.datasets import make_moons; X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)`</code><button class="copy-btn" onclick="copyCode(this)">Copy</button></td>
                </tr>                                                      <tr>
                    <td></td>
                    <td><code>ShuffleSplit</code></td>
                    <td>隨機分割數據集，用於交叉驗證。</td>
                    <td><code>n_splits</code>, <code>train_size</code>, <code>test_size</code>, <code>random_state</code>。</td>
                    <td class="code-snippet"><code>`from sklearn.model_selection import ShuffleSplit; ss = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)`</code><button class="copy-btn" onclick="copyCode(this)">Copy</button></td>
                </tr>''',
    '7': '''                <tr>
                    <td><strong>(Decision Trees & Ensembles)</strong></td>
                    <td><code>RandomForestClassifier</code>，<code>BaggingClassifier</code>，<code>VotingClassifier</code></td>
                    <td><strong>隨機森林</strong>：通過 <strong>Bagging</strong> 結合多棵決策樹，有效<strong>降低變異性</strong>。<code>VotingClassifier</code> 結合不同模型的預測結果。</td>
                    <td><strong><code>n_estimators</code></strong> (樹的數量)；<strong><code>max_features</code></strong> (每次分割時隨機採樣的特徵數)；<strong><code>voting</code></strong> 類型 ('hard' 或 'soft')。</td>
                    <td class="code-snippet"><code>`from sklearn.ensemble import RandomForestClassifier; rf = RandomForestClassifier(n_estimators=100).fit(X, y)`</code><button class="copy-btn" onclick="copyCode(this)">Copy</button></td>
                </tr>                                                                                                                                <tr>
                    <td></td>
                    <td><code>GradientBoostingRegressor</code>/<code>Classifier</code></td>
                    <td><strong>梯度提升</strong> (Boosting)：序列訓練，每一步都糾正前一個模型的<strong>殘差誤差</strong>，以<strong>降低偏差</strong>。</td>
                    <td><strong><code>learning_rate</code></strong> (學習率，用於收縮), <strong><code>n_estimators</code></strong> (樹的數量), <code>max_depth</code>。</td>
                    <td class="code-snippet"><code>`from sklearn.ensemble import GradientBoostingClassifier; gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1).fit(X, y)`</code><button class="copy-btn" onclick="copyCode(this)">Copy</button></td>
                </tr>                                                                                                                 <tr>
                    <td></td>
                    <td><code>AdaBoostClassifier</code></td>
                    <td>AdaBoost 分類器。</td>
                    <td><code>n_estimators</code>, <code>learning_rate</code>, <code>algorithm</code>。</td>
                    <td class="code-snippet"><code>`from sklearn.ensemble import AdaBoostClassifier; ada = AdaBoostClassifier(n_estimators=50).fit(X, y)`</code><button class="copy-btn" onclick="copyCode(this)">Copy</button></td>
                </tr>                                                             <tr>
                    <td></td>
                    <td><code>ExtraTreesClassifier</code></td>
                    <td>極端隨機樹分類器。</td>
                    <td><code>n_estimators</code>, <code>criterion</code>, <code>max_depth</code>。</td>
                    <td class="code-snippet"><code>`from sklearn.ensemble import ExtraTreesClassifier; et = ExtraTreesClassifier(n_estimators=100).fit(X, y)`</code><button class="copy-btn" onclick="copyCode(this)">Copy</button></td>
                </tr>                                                          <tr>
                    <td></td>
                    <td><code>HistGradientBoostingRegressor</code></td>
                    <td>基於直方圖的梯度提升迴歸樹。</td>
                    <td><code>learning_rate</code>, <code>max_iter</code>, <code>max_leaf_nodes</code>。</td>
                    <td class="code-snippet"><code>`from sklearn.ensemble import HistGradientBoostingRegressor; hgb = HistGradientBoostingRegressor(max_iter=100).fit(X, y)`</code><button class="copy-btn" onclick="copyCode(this)">Copy</button></td>
                </tr>                   <tr>
                    <td></td>
                    <td><code>StackingClassifier</code></td>
                    <td>堆疊估計器。</td>
                    <td><code>estimators</code>, <code>final_estimator</code>, <code>cv</code>。</td>
                    <td class="code-snippet"><code>`from sklearn.ensemble import StackingClassifier; from sklearn.linear_model import LogisticRegression; stack = StackingClassifier(estimators=[(&#x27;rf&#x27;, RandomForestClassifier()), (&#x27;gb&#x27;, GradientBoostingClassifier())], final_estimator=LogisticRegression())`</code><button class="copy-btn" onclick="copyCode(this)">Copy</button></td>
                </tr>''',
    '8': '''                <tr>
                    <td><strong>降維與流形學習</strong></td>
                    <td><code>PCA</code>，<code>RandomizedPCA</code>，<code>KernelPCA</code></td>
                    <td><strong>主成分分析</strong> (PCA)：<strong>最流行降維演算法</strong>。通過 <strong>SVD</strong> 找出保留<strong>最大變異數</strong>的超平面。</td>
                    <td><strong><code>n_components</code></strong> (目標維度或解釋變異數比例，如 95%)。<code>svd_solver="randomized"</code> (用於加速)。</td>
                    <td class="code-snippet"><code>`from sklearn.decomposition import PCA; pca = PCA(n_components=2).fit(X); X_pca = pca.transform(X)`</code><button class="copy-btn" onclick="copyCode(this)">Copy</button></td>
                </tr>                                                                                                                     <tr>
                    <td><strong>(Dimensionality Reduction)</strong></td>
                    <td><code>LocallyLinearEmbedding</code> (LLE)</td>
                    <td><strong>流形學習</strong>：用於處理<strong>非線性數據集</strong>（如 Swiss Roll）。</td>
                    <td><code>n_neighbors</code> (k-nearest neighbors)；<code>max_iter</code>。</td>
                    <td class="code-snippet"><code>`from sklearn.manifold import LocallyLinearEmbedding; lle = LocallyLinearEmbedding(n_neighbors=10, n_components=2).fit_transform(X)`</code><button class="copy-btn" onclick="copyCode(this)">Copy</button></td>
                </tr>                                                                   <tr>
                    <td></td>
                    <td><code>IncrementalPCA</code></td>
                    <td>增量主成分分析。</td>
                    <td><code>n_components</code>, <code>whiten</code>。</td>
                    <td class="code-snippet"><code>`from sklearn.decomposition import IncrementalPCA; ipca = IncrementalPCA(n_components=2).fit(X)`</code><button class="copy-btn" onclick="copyCode(this)">Copy</button></td>
                </tr>                                                                                                       <tr>
                    <td></td>
                    <td><code>LinearDiscriminantAnalysis</code></td>
                    <td>線性判別分析。</td>
                    <td><code>solver</code>, <code>shrinkage</code>, <code>n_components</code>。</td>
                    <td class="code-snippet"><code>`from sklearn.discriminant_analysis import LinearDiscriminantAnalysis; lda = LinearDiscriminantAnalysis(n_components=2).fit(X, y)`</code><button class="copy-btn" onclick="copyCode(this)">Copy</button></td>
                </tr>                                   <tr>
                    <td></td>
                    <td><code>Isomap</code></td>
                    <td>Isomap 嵌入。</td>
                    <td><code>n_neighbors</code>, <code>n_components</code>。</td>
                    <td class="code-snippet"><code>`from sklearn.manifold import Isomap; iso = Isomap(n_neighbors=10, n_components=2).fit_transform(X)`</code><button class="copy-btn" onclick="copyCode(this)">Copy</button></td>
                </tr>                                                                                                         <tr>
                    <td></td>
                    <td><code>MDS</code></td>
                    <td>多維縮放。</td>
                    <td><code>n_components</code>, <code>metric</code>。</td>
                    <td class="code-snippet"><code>`from sklearn.manifold import MDS; mds = MDS(n_components=2).fit_transform(X)`</code><button class="copy-btn" onclick="copyCode(this)">Copy</button></td>
                </tr>                                                                                                                                          <tr>
                    <td></td>
                    <td><code>TSNE</code></td>
                    <td>t-分佈隨機鄰域嵌入。</td>
                    <td><code>n_components</code>, <code>perplexity</code>, <code>learning_rate</code>。</td>
                    <td class="code-snippet"><code>`from sklearn.manifold import TSNE; tsne = TSNE(n_components=2, perplexity=30).fit_transform(X)`</code><button class="copy-btn" onclick="copyCode(this)">Copy</button></td>
                </tr>                                                                             <tr>
                    <td></td>
                    <td><code>GaussianRandomProjection</code>, <code>SparseRandomProjection</code></td>
                    <td>隨機投影。</td>
                    <td><code>n_components</code>, <code>eps</code>。</td>
                    <td class="code-snippet"><code>`from sklearn.random_projection import GaussianRandomProjection; grp = GaussianRandomProjection(n_components=2).fit_transform(X)`</code><button class="copy-btn" onclick="copyCode(this)">Copy</button></td>
                </tr>                                <tr>
                    <td></td>
                    <td><code>johnson_lindenstrauss_min_dim</code></td>
                    <td>找到安全的隨機投影維度。</td>
                    <td><code>n_samples</code>, <code>eps</code>。</td>
                    <td class="code-snippet"><code>`from sklearn.random_projection import johnson_lindenstrauss_min_dim; min_dim = johnson_lindenstrauss_min_dim(n_samples=1000, eps=0.1)`</code><button class="copy-btn" onclick="copyCode(this)">Copy</button></td>
                </tr>''',
    '10': '''                <tr>
                    <td></td>
                    <td><code>MLPClassifier</code></td>
                    <td>多層感知器分類器。</td>
                    <td><code>hidden_layer_sizes</code>, <code>activation</code>, <code>solver</code>。</td>
                    <td class="code-snippet"><code>`from sklearn.neural_network import MLPClassifier; mlp = MLPClassifier(hidden_layer_sizes=(100,)).fit(X, y)`</code><button class="copy-btn" onclick="copyCode(this)">Copy</button></td>
                </tr>                                                           <tr>
                    <td><strong>深度學習核心組件</strong></td>
                    <td><code>tf.keras.models.Sequential</code>，<code>tf.keras.layers.Dense</code></td>
                    <td><strong>序列模型</strong> (Sequential API)：構建 MLP 網路。<strong>全連接層</strong>。</td>
                    <td><code>units</code> (神經元數量)，<code>activation</code> (激活函數)，<code>kernel_initializer</code>。</td>
                    <td class="code-snippet"><code>`import tensorflow as tf; model = tf.keras.Sequential([tf.keras.layers.Dense(10, activation=&#x27;relu&#x27;)])`</code><button class="copy-btn" onclick="copyCode(this)">Copy</button></td>
                </tr>                            <tr>
                    <td></td>
                    <td><code>Perceptron</code>, <code>MLPClassifier</code>, <code>MLPRegressor</code></td>
                    <td>感知器和多層感知器。</td>
                    <td><code>hidden_layer_sizes</code>, <code>activation</code>, <code>solver</code>。</td>
                    <td class="code-snippet"><code>`from sklearn.neural_network import MLPClassifier; mlp = MLPClassifier(hidden_layer_sizes=(100,)).fit(X, y)`</code><button class="copy-btn" onclick="copyCode(this)">Copy</button></td>
                </tr>''',
    '11': '''                <tr>
                    <td><strong>(Deep Learning Core Components)</strong></td>
                    <td><code>tf.keras.layers.BatchNormalization</code></td>
                    <td><strong>批次正規化</strong> (BN)：<strong>加速訓練</strong>並緩解<strong>梯度不穩定性</strong>。</td>
                    <td><code>momentum</code> (用於移動平均)；<code>axis</code> (決定正規化 的維度)。</td>
                    <td class="code-snippet"><code>`model.add(tf.keras.layers.BatchNormalization())`</code><button class="copy-btn" onclick="copyCode(this)">Copy</button></td>
                </tr>                                                                                                                        <tr>
                    <td></td>
                    <td><code>tf.keras.layers.Dropout</code>，<code>tf.keras.regularizers.l1</code>/<code>l2</code></td>
                    <td><strong>正則化技術</strong>：<strong>Dropout</strong> 減少神經元共適應，提升泛化能力。L1/L2 懲罰項約束權重。</td>
                    <td><strong><code>rate</code></strong> (丟棄的機率，例如 0.5)；<code>factor</code> (正規化強度)。</td>
                    <td class="code-snippet"><code>`model.add(tf.keras.layers.Dropout(0.5))`</code><button class="copy-btn" onclick="copyCode(this)">Copy</button></td>
                </tr>''',
    '12': '''                <tr>
                    <td><strong>TensorFlow 底層與客製化</strong></td>
                    <td><code>tf.GradientTape</code>，<code>@tf.function</code></td>
                    <td><strong>自動微分</strong> (Autodiff)：記錄計算過程以<strong>計算梯度</strong>。使用 <code>@tf.function</code> 將 Python 函數轉換為 <strong>TensorFlow Graph</strong> 進行優化。</td>
                    <td>無 (用於控制梯度計算的上下文)。</td>
                    <td class="code-snippet"><code>`import tensorflow as tf; with tf.GradientTape() as tape: loss = compute_loss(); grads = tape.gradient(loss, variables)`</code><button class="copy-btn" onclick="copyCode(this)">Copy</button></td>
                </tr>                <tr>
                    <td><strong>(Low-Level TF & Customization)</strong></td>
                    <td><code>tf.Variable</code>，<code>tf.constant</code>，<code>tf.placeholder</code></td>
                    <td><strong>張量與操作</strong>：<code>Variable</code> (可變參數)；<code>constant</code> (常量)；<code>placeholder</code> (外部輸入)。</td>
                    <td>N/A。</td>
                    <td class="code-snippet"><code>`var = tf.Variable([1.0, 2.0]); const = tf.constant([3.0, 4.0])`</code><button class="copy-btn" onclick="copyCode(this)">Copy</button></td>
                </tr>''',
    '13': '''                <tr>
                    <td><strong>數據管道與特徵預處理</strong></td>
                    <td><code>tf.data.Dataset</code>，<code>tf.data.TextLineDataset</code>，<code>interleave()</code></td>
                    <td><strong>高效數據加載</strong>：處理大規模數據 (如 CSV, TFRecords)。使用 <code>interleave()</code> 進行<strong>並行讀取</strong>以提高效率 。</td>
                    <td><code>cycle_length</code> (並行讀取的文件數)；<code>batch_size</code>。</td>
                    <td class="code-snippet"><code>`dataset = tf.data.Dataset.from_tensor_slices((X, y)).batch(32)`</code><button class="copy-btn" onclick="copyCode(this)">Copy</button></td>
                </tr>                                <tr>
                    <td><strong>(Data Pipelines & Feature Prep)</strong></td>
                    <td><code>tf.keras.layers.Embedding</code>，<code>TextVectorization</code>，<code>CategoryEncoding</code></td>
                    <td><strong>特徵預處理層</strong>：將文本/類別特徵轉換為數值向量。<strong>詞嵌入</strong> (Word Embeddings) 處理高維文本。</td>
                    <td><strong><code>output_dim</code></strong> (Embedding 嵌入維度)；<code>output_mode</code> ("tf_idf", "multi_hot", "count")。</td>
                    <td class="code-snippet"><code>`embedding = tf.keras.layers.Embedding(input_dim=1000, output_dim=64)`</code><button class="copy-btn" onclick="copyCode(this)">Copy</button></td>
                </tr>                                                                                                                             <tr>
                    <td></td>
                    <td><code>tensorflow.train</code></td>
                    <td>用於創建 TFRecord 文件的工具。</td>
                    <td>N/A。</td>
                    <td class="code-snippet"><code>`writer = tf.io.TFRecordWriter(&#x27;data.tfrecord&#x27;)`</code><button class="copy-btn" onclick="copyCode(this)">Copy</button></td>
                </tr>                     <tr>
                    <td></td>
                    <td><code>tensorflow_datasets</code></td>
                    <td>數據集集合，可與 TensorFlow 一起使用。</td>
                    <td>N/A。</td>
                    <td class="code-snippet"><code>`import tensorflow_datasets as tfds; ds = tfds.load(&#x27;mnist&#x27;)`</code><button class="copy-btn" onclick="copyCode(this)">Copy</button></td>
                </tr>                                                                                                                                                <tr>
                    <td></td>
                    <td><code>tensorflow_hub</code></td>
                    <td>訓練過的機器學習模型倉庫。</td>
                    <td>N/A。</td>
                    <td class="code-snippet"><code>`import tensorflow_hub as hub; model = hub.load(&#x27;https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/4&#x27;)`</code><button class="copy-btn" onclick="copyCode(this)">Copy</button></td>
                </tr>''',
    '14': '''                <tr>
                    <td><strong>深度電腦視覺</strong></td>
                    <td><code>Conv2D</code>, <code>MaxPooling2D</code>，<code>Conv2DTranspose</code></td>
                    <td><strong>卷積神經網路</strong> (CNN) 基礎：<strong>卷積層</strong>提取局部特徵，<strong>池化層</strong>進行降採樣。</td>
                    <td><strong><code>filters</code></strong> ( 濾鏡數量), <strong><code>kernel_size</code></strong> (核心尺寸), <strong><code>strides</code></strong> (步幅)；<code>padding</code> ("VALID" 或 "SAME")。</td>
                    <td class="code-snippet"><code>`model.add(tf.keras.layers.Conv2D(32, (3, 3), activation=&#x27;relu&#x27;))`</code><button class="copy-btn" onclick="copyCode(this)">Copy</button></td>
                </tr>                                                                                                <tr>
                    <td><strong>(Deep Computer Vision)</strong></td>
                    <td><code>ResNet</code>, <code>YOLO</code> (架構)</td>
                    <td><strong>進階 CNN 架構</strong>：<strong>ResNet</strong> 使用 <strong>Skip Connections</strong> 解決深層網路梯度問題。<strong>YOLO</strong> (You Only Look Once) 應用於<strong>物體偵測</strong> (Object Detection)。</td>
                    <td><strong>遷移學習</strong>中的<strong>凍結層</strong> (<code>Trainable=False</code>)。</td>
                    <td class="code-snippet"><code>`base_model = tf.keras.applications.ResNet50(weights=&#x27;imagenet&#x27;, include_top=False)`</code><button class="copy-btn" onclick="copyCode(this)">Copy</button></td>
                </tr>                                                                                                             <tr>
                    <td></td>
                    <td><code>tensorflow_datasets</code></td>
                    <td>數據集集合。</td>
                    <td>N/A。</td>
                    <td class="code-snippet"><code>`import tensorflow_datasets as tfds; ds = tfds.load(&#x27;cifar10&#x27;)`</code><button class="copy-btn" onclick="copyCode(this)">Copy</button></td>
                </tr>''',
    '15': '''                <tr>
                    <td><strong>序列處理與 RNNs/Attention</strong></td>
                    <td><code>LSTM</code>，<code>GRU</code>，<code>SimpleRNN</code></td>
                    <td><strong>循環神經網路</strong> (RNNs)：用於<strong>時間序列預測</strong>。<strong>LSTM</strong> 和 <strong>GRU</strong> 解決梯度消失/長期依賴問題。</td>
                    <td><strong><code>units</code></strong> (隱藏單元數量)；<code>return_sequences</code>；<code>stateful</code> (有狀態 RNN)。</td>
                    <td class="code-snippet"><code>`model.add(tf.keras.layers.LSTM(64, return_sequences=True))`</code><button class="copy-btn" onclick="copyCode(this)">Copy</button></td>
                </tr>           <tr>
                    <td></td>
                    <td><code>statsmodels</code></td>
                    <td>統計建模和計量經濟學庫。</td>
                    <td>N/A。</td>
                    <td class="code-snippet"><code>`import statsmodels.api as sm; model = sm.OLS(y, X).fit()`</code><button class="copy-btn" onclick="copyCode(this)">Copy</button></td>
                </tr>''',
    '16': '''                <tr>
                    <td><strong>(Sequences, RNNs & Attention)</strong></td>
                    <td><code>MultiHeadAttention</code>，<code>Transformer</code> (架構)，<code>Positional Encoding</code></td>
                    <td><strong>注意力機制</strong> (Attention)：用於 <strong>NMT</strong> (機器翻譯)。<strong>Transformer</strong> 架構完全基於注意力，通過<strong>位置編碼</strong> (Positional Encoding) 引入序列順序。</td>
                    <td><strong><code>num_heads</code></strong> (Attention 頭數)；Beam Search <strong><code>beam width</code></strong> (用於推論)。</td>
                    <td class="code-snippet"><code>`attention = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=64)`</code><button class="copy-btn" onclick="copyCode(this)">Copy</button></td>
                </tr>''',
    '17': '''                <tr>
                    <td><strong>(Unsupervised Learning)</strong></td>
                    <td><code>Autoencoder</code></td>
                    <td><strong>自動編碼器</strong>：用於降維、特徵提取、<strong>生成式模型</strong> (Generative Models)。</td>
                    <td><strong><code>latent_dim</code></strong> (編碼層維度) ；<strong>權重綁定</strong> (Tying Weights)；<code>reconstruction_loss</code>。</td>
                    <td class="code-snippet"><code>`import tensorflow as tf; encoder = tf.keras.Sequential([tf.keras.layers.Dense(latent_dim, activation=&#x27;relu&#x27;)])`</code><button class="copy-btn" onclick="copyCode(this)">Copy</button></td>
                </tr>                    <tr>
                    <td><strong>生成式模型</strong></td>
                    <td><code>Autoencoder</code>，<code>GAN</code> (Generator/Discriminator)，<code>DDPM</code> (Diffusion Models)</td>
                    <td><strong>自動編碼器</strong> (Autoencoders)：用於數據壓縮。<strong>VAE</strong> (變分自動編碼器)；<strong>生成式對抗網路</strong> (GANs)；<strong>擴散模型</strong> (DDPM) 用於高品質圖像生成。</td>
                    <td><strong><code>latent_dim</code></strong> (潛在空間維度)；<strong>損失函數</strong> (e.g., VAE 的 KL 散度項)；訓練<strong>生成器和判別器的平衡</strong> (GANs)。</td>
                    <td class="code-snippet"><code>`generator = tf.keras.Sequential([tf.keras.layers.Dense(128, activation=&#x27;relu&#x27;), tf.keras.layers.Dense(latent_dim)])`</code><button class="copy-btn" onclick="copyCode(this)">Copy</button></td>
                </tr>''',
    '18': '''                <tr>
                    <td><strong>強化學習</strong></td>
                    <td><code>gym.make()</code>, <code>Policy Gradients</code>, <code>DQN</code> (Deep Q-Network)</td>
                    <td><strong>環境互動</strong> (如 <code>CartPole</code>)。<strong>策略梯度</strong> (Policy Gradients, PG)；<strong>深度 Q 網路</strong> (DQN) 及其變體。</td>
                    <td><strong><code>discount_factor</code> ($\gamma$)</strong>；<strong>探索策略</strong> ($\epsilon$, for $\epsilon$-greedy)；<code>learning_rate</code>。</td>
                    <td class="code-snippet"><code>`import gym; env = gym.make(&#x27;CartPole-v1&#x27;); obs = env.reset(); action = env.action_space.sample()`</code><button class="copy-btn" onclick="copyCode(this)">Copy</button></td>
                </tr>                                                                <tr>
                    <td><strong>(Reinforcement Learning, RL)</strong></td>
                    <td><code>env.step()</code>, <code>env.reset()</code></td>
                    <td><strong>環境操作</strong>：<code>step()</code> 執行動作並返回新的觀測、獎勵和狀態；<code>reset()</code> 進行環境初始化。</td>
                    <td>N/A。</td>
                    <td class="code-snippet"><code>`obs, reward, done, info = env.step(action); obs = env.reset()`</code><button class="copy-btn" onclick="copyCode(this)">Copy</button></td>
                </tr>        <tr>
                    <td></td>
                    <td><code>gymnasium</code></td>
                    <td>開發和比較強化學習演算法的工具包。</td>
                    <td>N/A。</td>
                    <td class="code-snippet"><code>`import gymnasium as gym; env = gym.make(&#x27;CartPole-v1&#x27;)`</code><button class="copy-btn" onclick="copyCode(this)">Copy</button></td>
                </tr>''',
    '19': '''                <tr>
                    <td><strong>規模化與部署</strong></td>
                    <td><code>tf.distribute.MirroredStrategy</code>，<code>MultiWorkerMirroredStrategy</code></td>
                    <td><strong>分散式訓練</strong>：在多 GPU 或多台機器上進行<strong>數據并行化</strong> (Data Parallelism)，以加速大規模模型的訓練。</td>
                    <td><strong><code>batch_size</code></strong> (需能被副本數整除)。</td>
                    <td class="code-snippet"><code>`strategy = tf.distribute.MirroredStrategy(); with strategy.scope(): model = create_model()`</code><button class="copy-btn" onclick="copyCode(this)">Copy</button></td>
                </tr>                                            <tr>
                    <td><strong>(Scaling & Deployment)</strong></td>
                    <td><code>TFLiteConverter</code>，<code>tf.saved_model.save</code>，<code>Keras Tuner</code></td>
                    <td><strong>部署與優化</strong>：將模型轉換為 TFLite 格式以部署到行動/嵌入式設備；使用 <strong>Keras Tuner</strong> 進行超參數自動優化。</td>
                    <td>超參數調優的<strong>搜索範圍</strong> (Search Space)；<code>max_trial_count</code> (最大嘗試次數)。</td>
                    <td class="code-snippet"><code>`converter = tf.lite.TFLiteConverter.from_saved_model(&#x27;saved_model_path&#x27;); tflite_model = converter.convert()`</code><button class="copy-btn" onclick="copyCode(this)">Copy</button></td>
                </tr>                                                                                                          <tr>
                    <td></td>
                    <td><code>google.cloud.aiplatform</code></td>
                    <td>Vertex AI 用於機器學習。</td>
                    <td>N/A。</td>
                    <td class="code-snippet"><code>`from google.cloud import aiplatform; aiplatform.init(project=&#x27;my-project&#x27;)`</code><button class="copy-btn" onclick="copyCode(this)">Copy</button></td>
                </tr>                                                                                                                                           <tr>
                    <td></td>
                    <td><code>hypertune</code></td>
                    <td>超參數調優庫。</td>
                    <td>N/A。</td>
                    <td class="code-snippet"><code>`import hypertune; hpt = hypertune.HyperTune()`</code><button class="copy-btn" onclick="copyCode(this)">Copy</button></td>
                </tr>                                                       <tr>
                    <td></td>
                    <td><code>keras_tuner</code></td>
                    <td>Keras 模型的超參數調優。</td>
                    <td>N/A。</td>
                    <td class="code-snippet"><code>`import keras_tuner as kt; tuner = kt.Hyperband(build_model, objective=&#x27;val_accuracy&#x27;)`</code><button class="copy-btn" onclick="copyCode(this)">Copy</button></td>
                </tr>                                                                                                                                            <tr>
                    <td></td>
                    <td><code>tensorboard</code></td>
                    <td>TensorFlow 可視化工具包。</td>
                    <td>N/A。</td>
                    <td class="code-snippet"><code>`import tensorboard; %tensorboard --logdir logs`</code><button class="copy-btn" onclick="copyCode(this)">Copy</button></td>
                </tr>                                         <tr>
                    <td></td>
                    <td><code>tqdm</code></td>
                    <td>進度條庫。</td>
                    <td>N/A。</td>
                    <td class="code-snippet"><code>`from tqdm import tqdm; for i in tqdm(range(100)): pass`</code><button class="copy-btn" onclick="copyCode(this)">Copy</button></td>
                </tr>                                                       <tr>
                    <td></td>
                    <td><code>transformers</code></td>
                    <td>最新的自然語言處理庫。</td>
                    <td>N/A。</td>
                    <td class="code-snippet"><code>`from transformers import pipeline; classifier = pipeline(&#x27;sentiment-analysis&#x27;)`</code><button class="copy-btn" onclick="copyCode(this)">Copy</button></td>
                </tr> <tr>
                    <td></td>
                    <td><code>PIL</code></td>
                    <td>Python 圖像處理庫。</td>
                    <td>N/A。</td>
                    <td class="code-snippet"><code>`from PIL import Image; img = Image.open(&#x27;image.jpg&#x27;)`</code><button class="copy-btn" onclick="copyCode(this)">Copy</button></td>
                </tr>                                       <tr>
                    <td></td>
                    <td><code>nltk</code></td>
                    <td>自然語言處理工具包。</td>
                    <td>N/A。</td>
                    <td class="code-snippet"><code>`import nltk; nltk.download(&#x27;punkt&#x27;)`</code><button class="copy-btn" onclick="copyCode(this)">Copy</button></td>
                </tr>                                                      <tr>
                    <td></td>
                    <td><code>graphviz</code></td>
                    <td>圖形可視化軟體。</td>
                    <td>N/A。</td>
                    <td class="code-snippet"><code>`import graphviz; dot = graphviz.Digraph()`</code><button class="copy-btn" onclick="copyCode(this)">Copy</button></td>
                </tr>'''
}

# Function to replace table content for each chapter
for chapter_num, new_rows in chapter_data.items():
    # Find the section for the chapter
    section_pattern = rf'<!-- Chapter {chapter_num}: .*? -->\s*<section id="chapter-{chapter_num}">\s*<h2 class="section-title">第{chapter_num}章：.*?</h2>\s*<!-- .*? -->\s*<div class="card">\s*<table>\s*<tr>\s*<th>關鍵概念</th>\s*<th>重要函數</th>\s*<th>功能簡介與應用場景</th>\s*<th>Fine-tune 用參數</th>\s*<th>範例代碼片段</th>\s*</tr>'
    section_match = re.search(section_pattern, html_content, re.DOTALL)
    if section_match:
        section_start = section_match.start()
        # Find the end of the table
        table_end_pattern = r'</table>\s*</div>\s*</section>'
        table_end_match = re.search(table_end_pattern, html_content[section_start:], re.DOTALL)
        if table_end_match:
            table_end = section_start + table_end_match.start()
            # Extract the old table content
            old_table = html_content[section_start:table_end]
            # Find the header row
            header_end = old_table.find('</tr>') + 5
            # Replace the rows after header
            new_table = old_table[:header_end] + new_rows + '\n            </table>'
            html_content = html_content[:section_start] + new_table + html_content[table_end:]
            print(f"Replaced chapter {chapter_num}")

# Write back the updated HTML
with open('機器學習與深度學習實戰速查手冊.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print("All chapters updated successfully!")