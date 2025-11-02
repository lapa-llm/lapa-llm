
# install llama.cpp with CUDA support

```
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DLLAMA_CURL=OFF -j 8
cmake --build build --config Release -j 8
```

```
cd ./models
hf download lapa-llm/lapa-v0.1.2-instruct --local-dir ./lapa-12b-it
hf download lapa-llm/tokenizer --local-dir ./lapa-12b-it --include uk_tokenizer.model
mv ./lapa-12b-it/uk_tokenizer.model ./lapa-12b-it/tokenizer.model

cd ..

# install Python dependencies
python3 -m pip install -r requirements.txt

# convert the model to ggml FP16 format
python3 convert_hf_to_gguf.py ./models/lapa-12b-it/

# TODO: use llama-matrix for calibration
# ./build/bin/llama-imatrix

# quantize the model to 4-bits (using Q4_K_M method)
./build/bin/llama-quantize ./models/lapa-12b-it/lapa-12B-it-F16.gguf ./models/lapa-12b-it/lapa-v0.1.2-instruct-Q4_K_M.gguf Q4_K_M

./build/bin/llama-quantize ./models/lapa-12b-it/lapa-12B-it-F16.gguf ./models/lapa-12b-it/lapa-v0.1.2-instruct-Q4_K_S.gguf Q4_K_S

./build/bin/llama-quantize ./models/lapa-12b-it/lapa-12B-it-F16.gguf ./models/lapa-12b-it/lapa-v0.1.2-instruct-Q5_K_S.gguf Q5_K_S

./build/bin/llama-quantize ./models/lapa-12b-it/lapa-12B-it-F16.gguf ./models/lapa-12b-it/lapa-v0.1.2-instruct-Q8_0.gguf Q8_0

```