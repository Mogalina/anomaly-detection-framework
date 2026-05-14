# TODOs

## Thesis Subjects To Be Covered

* Edge Computing
    * Metrics collection
        * What metrics to collect?
        * How to collect metrics?
* Long Short Term Memory (LSTM)
    * Architecture
    * Recurrent Neural Network (RNN) vs LSTM
    * Why LSTM?
* Autoencoders
    * Architecture
    * Why Autoencoders?
    * Autoencoders types
        * Denoising Autoencoders
        * Contractive Autoencoders
        * Sparse Autoencoders
        * Variational Autoencoders
    * Autoencoders vs Principal Component Analysis (PCA)
* Autoencoders (AE) + Long Short Term Memory (LSTM)
    * Architecture
    * Why Autoencoders + LSTM?
    * Current architecture description (Input -> Encoder LSTM -> Latent space -> Decoder LSTM -> Linear layer -> Reconstructed sequence)
        * Parameters count
        * Latent space size (why)
        * Model footprint
        * Model inference time
        * Model inference latency
    * Why Linear layer?
    * Why Reconstructed sequence?
    * Why Latent space?
    * Why Encoder LSTM?
    * Why Decoder LSTM?
    * Bidirectional LSTM
        * Past -> Future
        * Future -> Past
* Federated Learning
    * Architecture
        * Aggregation Strategies
        * FedAvg
    * Why Federated Learning?
        * Security
        * Privacy
        * Cost
    * How to implement Federated Learning?
    * How to train model?
    * How to evaluate model?
    * How to deploy model?
    * Differential Privacy
        * Why Differential Privacy?
        * How DP works (noise addition, gradient clipping, etc.)?
        * Why DP in Federated Learning / Edge Computing / Distributed Systems?
        * How does DP ensure accurate model training?
        * How does DP ensure privacy?
* Message Serialization
    * Types
        * JSON
        * Protobuf
        * FlatBuffers
    * Why Protobuf/FlatBuffers?
* Message Sending
    * Types
        * TCP
        * UDP
    * Why TCP?
    * Why UDP?
    * Compression
        * LZ4
        * Zstd
    * LZ4 vs Zstd vs ...
        * Speed
        * Compression ratio
        * Memory usage
        * CPU usage
        * Latency
    * Why LZ4?
* Root Cause Analysis (RCA)
    * Motivation
    * Dependency Graph / Causal Graph
    * Trace Collector
    * Types
        * PageRank
        * Classification
    * Why PageRank?
    * Why Classification?
    * Implementation
* Adaptive Thresholding
    * Motivation
        * Adaptive vs Fixed Thresholding
    * How it works
    * Implementation
        * Per service customization
    * Reinforcement Learning
    * Q-learning
        * Implementations
        * Advantages
        * Disadvantages
* Reliability
    * CAP Theorem
    * Single Point of Failure (SPoF)
    * Redundancy
* Adaptive message compression over the network
    * Compression Algorithms
    * LZ4 vs Zstd
    * Compression ratio
    * Speed
    * Memory usage
    * CPU usage
    * Latency
    * Why Zstd is usually better for Federated Learning
    * Why LZ4 might still be chosen
    * Adaptive compression
        * Adaptive vs Fixed Compression
        * How it works
        * Implementation
            * Edge Device (Node)
            * Central Coordinator
* Fault tolerance and recovery
    * Adaptive node profiles (in federated learning)
        * Lightweight
            * stateless, no local writes, minimal deps. Joins → trains → submits → disconnects. Recovers by re-registering
        * Standard
            * opt-in for nodes with persistent storage. Caches last update in SQLite so it can replay after a crash
* Future Improvements
    * Transfer Learning

## Fixes

* Persistent memory
    * Persistent casual graph 
    * Persistent anomaly detection model
    * Persistent federated learning model
    * Persistent root cause analysis model
* Federated Learning communication
    * Communication methods over the network
        * gRPC
        * HTTP
    * gRPC communication between coordinator and edge nodes
    * Message Serialization
        * Protobuf
        * FlatBuffers

---

Advantages of LZ4 (over JSON)
Massive Reduction in Bandwidth JSON is a text-based format. Encoding a 32-bit floating-point number (e.g., 0.12345678) into JSON turns it into an 10-byte string. A binary format reduces this to 4 bytes, and applying LZ4 compression on structured, repetitive gradients or sparse tensors can compress this data significantly further, reducing network congestion.

Extremely High Speed (Low Overhead) LZ4 is specifically optimized for speed, not just the compression ratio. It can compress at >500 MB/s and decompress at multiple GB/s per CPU core. In FL, where edge devices might have limited resources (like Raspberry Pis or lightweight VMs), LZ4 adds almost zero latency or CPU tax to the pipeline compared to heavy parsers. JSON stringification/parsing in Python is notoriously slow for massive arrays.

Lower Memory Footprint Processing large JSON payloads requires loading the entire string array into RAM, parsing it, and instantiating Python/JSON objects before finally converting it back to a PyTorch/NumPy array. LZ4 combined with a binary format (like MessagePack or Protobuf/gRPC) can be directly deserialized into memory buffers without massive memory spikes.

Disadvantages of LZ4
Lower Compression Ratio compared to GZIP/Zstd While LZ4 is incredibly fast, algorithm trade-offs mean it doesn't compress data as tightly as Gzip or Zstandard. If your network is extremely restricted (e.g., expensive satellite links) but your CPU has plenty of cycles to spare, Zstandard might actually be a better choice.

Debugging and Visibility JSON is human-readable. If a payload faults or if you want to inspect a request via a standard HTTP sniffer (like Wireshark or Postman), you can easily read JSON text. LZ4 payloads are opaque binary blobs.

Interoperability / Implementation Complexity JSON is universally supported by every language's standard library. To use LZ4, you generally need to implement a binary serialization format (MessagePack, Protobuf, or direct numpy.tobytes()), compress it with the lz4 library, and carefully unpack and reshape the tensor on the coordinator.

This switch alone will drastically reduce your FL round latency (fl_bandwidth_bytes and processing times) while saving significant CPU on your edge devices.

---

1. Protobuf/FlatBuffers Provide Structure and Schema
Sending raw binary chunks (like NumPy arrays) over the wire is fast, but it lacks metadata. In Federated Learning, you aren't just sending arrays; you are sending structured data such as:

* Client IDs
* Model version numbers
* Number of training samples processed
* Metadata for differential privacy or gradient compression
* And, of course, the actual weight tensors

Protobuf or FlatBuffers give you a strongly-typed schema to consistently deserialize this metadata across different nodes, languages, and architectures, ensuring backward and forward compatibility as your FL system evolves.

2. LZ4 Reduces the Payload Size
* While Protobuf and FlatBuffers are much more compact than JSON because they don't store field names explicitly, the bulk of your payload will be the multi-megabyte tensor arrays (the model gradients).
* Protobuf encodes numbers efficiently, but an array of thousands of floats is still going to be large.
* LZ4 excels at quickly scanning over these large binary arrays and compressing redundant sequences or zeroes (which are very common if your framework already uses the gradient_compression: method: "topk" setup seen in your config!).

3. The Synergy: Speed Meets Efficiency
* FlatBuffers + LZ4: FlatBuffers is designed for zero-copy deserialization. You can decompress the LZ4 payload directly into a memory buffer and instantly read the FlatBuffer structures without any parsing overhead. This is arguably the fastest possible path for ingesting tensor updates on a centralized coordinator.
* Protobuf/gRPC + LZ4: gRPC natively uses Protobuf and has built-in support for message compression (including LZ4 in many language implementations). This means the networking layer handles the LZ4 compression transparently, and your application code only deals with clean Protobuf objects.

Because you are doing Top-K gradient compression, a large portion of the gradient updates being sent back to the coordinator are likely sparse arrays (many zeros or repeating patterns). LZ4 thrives on this exact kind of data.

---

In the context of Federated Learning (FL) for your Anomaly Detection Framework, Zstandard (Zstd) is generally the better choice over LZ4 for modern architectures, although the "best" choice depends on the specific hardware limits of your edge nodes.

Here is a breakdown of why you might choose one over the other in your specific context:

Why Zstd is usually better for Federated Learning:
Superior Compression Ratio for Sparse Data In your 

config.yaml
, your FL setup uses gradient_compression: method: "topk". This means the updates sent to the coordinator are highly sparse (mostly zeroes). Zstd's entropy coding (Huffman coding) and dictionary capabilities are exceptionally good at compressing sparse arrays, achieving much higher compression ratios on gradients than LZ4.

Tunable Performance Zstd is highly tunable. It offers compression levels from 1 to 22 (and even negative levels for extra speed). At lower levels (e.g., Level 1), Zstd's compression speed approaches LZ4's, but it still yields a significantly smaller payload size.

Network vs. Compute Trade-off In Federated Learning, the network connection implies cellular, IoT, or unstable WAN links, making network bandwidth the primary bottleneck, not CPU speed. Your edge models (an LSTM with hidden_size: 64 and num_layers: 2) are relatively small. Zstd will compress these weights down to a tiny fraction of their original size, making the transmission almost instantaneous over constrained networks.

Why you might still choose LZ4:
Severely Constrained Edge Compute If your edge nodes are running on extremely weak hardware (e.g., low-power IoT microcontrollers with very limited CPU cycles and MBs of RAM), Zstd might consume too much CPU power or memory to compress the payload. LZ4 requires almost zero CPU tax and has a tiny memory footprint.

Speed is the Only Metric If your edge nodes and coordinator are situated on the same high-speed local network (e.g., a massive factory floor with Gigabit Wi-Fi), and you don't care about network bandwidth, LZ4 will get the data packaged and unpacked a negligible fraction of a millisecond faster than Zstd.

---

Why Zstd beats LZ4 in this context
As mentioned earlier, you should choose Zstd over LZ4 for Federated Learning.

You are bandwidth-bound, not compute-bound: In Federated Learning, transmitting megabytes of weights over unpredictable networks (IoT, cellular, or general WAN) is your biggest bottleneck. Zstd's superior compression ratio will drastically speed up your sync times.
Top-K Sparsity: Your 

config.yaml
 uses gradient_compression: method: "topk". This means you are sending mostly zeroes and repetitive structures. Zstd natively uses dictionary compression and Huffman coding, which destroys LZ4 at squeezing sparse matrices into tiny payloads.
Tunability: You can set Zstd to a low level (e.g., Level 1) to get speeds nearly matching LZ4, but with much better compression.

---

Ch. 2 → Ch. 4 → Ch. 6 → Ch. 3 → Ch. 9 → (Ch. 5/7/8 theory sections) → Ch. 10 → Ch. 11 → Ch. 1 → Ch. 12

---

rm -rf $(biber --cache)

pdflatex -interaction=batchmode main.tex && biber main && pdflatex -interaction=batchmode main.tex && pdflatex -interaction=batchmode main.tex

---

- (2) user interface for the anomaly detection framework
- (3) manually test the whole framework
- [x] (5) fix q-learning hyper-parameters and metrics
    - include results in the paper
- (5) mention the selected features for anomaly detection and their importance
- [x] fix differential privacy hyper-parameters and metrics
    - include results in the paper
- (4) run the following prompt when claude is available
"""
You are a researcher working on a federated anomaly detection system. Please review the attached research paper and propose the following:

    1. **Include New Experiments**: Include new benchmarks and tests performed on Q-learning and differential privacy (including clipping). Also, re-compile the graphs and create new ones if needed, then create new plots and tables to include in the paper. Finally, refactor the conclusions and discussions related to these experiments to ensure they are consistent with the new results.

    2. **Selected Features**: Discuss the selected features for anomaly detection and their importance (e.g. CPU load, memory usage, network traffic, etc. what is present in the framework) and how they are used to train the anomaly detection model. (The selected features are: CPU load, memory usage, network traffic, etc. what is present in the framework)

    3. **Summarize Thesis**: Summarize docs/thesis-final version to a 55-60 pages LaTeX document (total PDF pages). Keep the academic style of the thesis. Do not exclude important information from the thesis. You can restructurize the thesis to make it more coherent, but do not remove any significant information. You can modify the content chapters structure to make them more compact if it helps, but keep the academic style and formatting. You are NOT able to modify the global main.tex file formatting. 

    4. **Rewrite Conclusions**: Rewrite the conclusions section to include the new results and discussions.

    5. **Make Thesis Esthetically Pleasing**: Make the thesis more aesthetically pleasing. Use more figures, tables, and diagrams to make the thesis more interesting (where possible). Make more spacing using pharagraphs, section headings, tables, lists, etc. Also make sure that the thesis is well-formatted and easy to read. You are not able to modify the global main.tex file, only the .tex files inside chapters/ folder. You must keep the thesis in 54-55 pages range, not more not less (including first pages, and references).

Make sure to make the final thesis version to be an academic and professional thesis, ready to be presented.
"""

I have updated the adaptive thresholding (q-learning) and differential privacy benchmarks. Update the paper with these new results in the evaluation, conslusion, differential privacy, and adaptive thresholding chapters/sections to reflect the new results.