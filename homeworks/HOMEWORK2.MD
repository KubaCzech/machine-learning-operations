## HW 2: Serving a Machine Learning Model as an API Service, Dynamic Batching Benchmark

The goal of this homework is to **serve the model you trained in HOMEWORK_1.MD as a REST or gRPC API** using a model-serving framework, and to **measure how much dynamic (adaptive) batching changes inference throughput and latency**.

You can use any of the following model-serving frameworks:
* **BentoML**
* **NVIDIA Triton Inference Server**
* **TorchServe**
* **LitServe / Lightning serve**, **Ray Serve**, or any similar model-serving tool

**Important:** Use the model you trained for **HOMEWORK_1.MD**.

### Example flow to complete the homework:
* Choose a model-serving framework and **check in its documentation whether it supports dynamic batching**. If possible, pick one with *adaptive* batching, i.e. one where both the batch size and the batching window (max wait time / queue delay) adapt to the traffic, not just a fixed batch size. Examples of the relevant settings:
  * BentoML: adaptive batching (`batchable=True`, `max_batch_size`, `max_latency_ms`)
  * Triton: `dynamic_batching` in `config.pbtxt` (`preferred_batch_size`, `max_queue_delay_microseconds`)
  * TorchServe: `batch_size` and `max_batch_delay` in the model config
  * Ray Serve: `@serve.batch` (`max_batch_size`, `batch_wait_timeout_s`)
* Export or package your trained model from HOMEWORK_1.MD so the serving framework can load it (e.g. a Lightning checkpoint, TorchScript, ONNX or a framework-specific model store).
* Create a service that runs inference with the model and exposes an API endpoint.
* Send an example request (cURL, Python `requests` or the tool's official client) and show the returned prediction.
* Benchmark the service **without** dynamic batching and **with** dynamic batching, using the methodology described below.

### Benchmark methodology (required for everyone):
A single request timed with `time curl ...` is **not** a benchmark. To compare configurations fairly:

* **Primary metric: throughput** in **predictions per second** (samples/s), measured over a **fixed 60-second window** under a fixed number of concurrent clients. Report it as samples/s, not "requests per 10 s", so that results are comparable across people.
* **Secondary metrics: latency** p50 / p95 / p99 in ms. Batching usually raises throughput and can raise tail latency, so you need both numbers to judge whether it helps.
* **Warm-up:** discard the first ~10–20 s, or send a few hundred warm-up requests first. The first requests include model loading, CUDA context creation, JIT compilation, etc.
* **Concurrency sweep:** run at least **3 concurrency levels** (e.g. 1, 8, 32 concurrent clients). Dynamic batching does nothing at concurrency 1, so a single level can't show its effect.
* **Fixed conditions:** same hardware, same model, same input payload, same client machine for both configurations. Write down the hardware (CPU/GPU) and the batching parameters you used.
* **Repeat** each measurement at least 3 times and report the mean (and the standard deviation if you can).
* Suggested load-testing tools: Triton `perf_analyzer`, `locust`, `k6`, `hey`/`oha`, or your own `asyncio` + `httpx` script. Any of them is fine as long as you follow the rules above.

Report your results in a table like this:

| Concurrency | Throughput no batching [samples/s] | Throughput with batching [samples/s] | p95 latency no batching [ms] | p95 latency with batching [ms] |
|---|---|---|---|---|
| 1 | | | | |
| 8 | | | | |
| 32 | | | | |

### Additional organization information:
* If possible, please don't copy whole solutions outright.
* Please upload your code to the same repository as HOMEWORK_1 (or a new one, linked from it).
* If your chosen tool does **not** support dynamic batching, say so and justify the choice, or switch to one that does.

### Acceptance Criteria:
* **Model-serving configuration/code** is in the repository: e.g. a BentoML service file, a TorchServe setup, a Lightning serve script or a Triton model repository directory, plus the command used to start the service.
* **Client request example** using cURL, Python `requests` or a framework-specific client (e.g. `tritonclient`).
* **Working output:** show that the service returns a valid model prediction.
* **Benchmark** of your serving tool **with and without dynamic batching**, following the methodology above (the script/command used for load testing must be in the repository, together with the results table).
* If dynamic batching gives no improvement, that is also acceptable, **but you have to justify it (to me in person, do not waste time writing any reports etc)** and show your measurements, the batching parameters you tried, and why, in your opinion, it didn't help (e.g. the model is too small, a CPU bottleneck, client-side limits, too low concurrency).

### Presentation Guidelines:
The acceptance criteria above apply to the whole group. Additionally, a small subset of students can present their results in front of the whole group.

Presenters must benchmark **at least two model-serving tools** (e.g. BentoML vs Triton, TorchServe vs LitServe) on the same model and hardware, and present a comparison table with these four columns:

| Concurrency | Tool A, no batching | Tool A, adaptive batching | Tool B, no batching | Tool B, adaptive batching |
|---|---|---|---|---|
| 1 | samples/s, p95 ms | samples/s, p95 ms | samples/s, p95 ms | samples/s, p95 ms |
| 8 | | | | |
| 32 | | | | |

5 minute presentation (no slides necessary). **Points below are obligatory to discuss in order to obtain the presentation point.** The presentation (screen sharing and your speech) must showcase:
* Your model and task (briefly, a reminder from HW1) and the serving tools you chose and why.
* The service code / configuration for each tool (where the batching is configured).
* A live or recorded request returning a prediction.
* The comparison table above and which tool + configuration you would pick for production, and why.
* **Find the knee:** sweep concurrency (e.g. 1, 2, 4, …, 128) and plot throughput vs. p95 latency for both tools. Where does the throughput plateau, and where does latency blow up?
* **Who is the bottleneck?** Show (with GPU/CPU utilization, a profiler or the tool's metrics endpoint) whether the time goes to the model, serialization/deserialization, Python overhead or the network.
* **Dynamic vs. adaptive batching:** be ready to explain the difference between dynamic batching (requests are grouped on the server, usually with a fixed max batch size and a fixed max wait time) and adaptive batching (the batch size and/or the batching window adjust automatically to the incoming traffic and observed latency), and how batching is implemented in the tool that performed best in your benchmark.
* Main problems encountered (for others to learn from your mistakes).

#### Tryhard section (optional, for presenters who want to go above&beyond, not obligatory to obtain points, just aura points):
Pick any of the challenges below:
* **Tune the window:** sweep the batching window (max wait / queue delay) and show the throughput vs. tail-latency trade-off. What window would you pick for a latency SLA of e.g. 100 ms p95?
* **What did the batcher actually do?** Log the batch sizes the server really formed under each concurrency level and show their distribution. Did the batcher ever reach `max_batch_size`, or was it mostly sending batches of 1–2?
* **Format wars:** compare the same tool serving the model as eager PyTorch vs. TorchScript / ONNX / TensorRT. Does the export matter more than batching?
* **Break it:** find the load at which the service starts dropping requests or timing out, and explain why.
* **Predict before you reveal:** before showing the table, ask the group to guess which tool won and by how much batching helped.
* **One-sentence takeaway:** finish with a single concrete lesson that others could apply when serving their own models.

### Grading:
* The homework yields **1 point**.
* The presentation yields **1 additional point** (2 points in total for students who present).

### Deadline:
* Classes on 23.11.2026.
* Submitting the homework late yields a one-time penalty of 0.25 points.

### Useful links:
* https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/tutorials/Quick_Deploy/PyTorch/README.html
* https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html
* https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_analyzer/README.html
* https://docs.bentoml.com/en/latest/examples/overview.html
* https://docs.bentoml.com/en/latest/get-started/adaptive-batching.html
* https://pytorch.org/serve/
* https://docs.ray.io/en/latest/serve/advanced-guides/dyn-req-batch.html
* https://locust.io/
