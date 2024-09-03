import React, { useState } from "react";

const Form: React.FC = () => {
  const [dataset, setDataset] = useState("mnist");
  const [arch, setArch] = useState([{ type: "", size: "" }]);
  const [epochs, setEpochs] = useState(10);

  const datasets = [
    {
      value: "mnist",
      label: "MNIST",
      imgSrc: "https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png",
    },
    {
      value: "fashion_mnist",
      label: "Fashion MNIST",
      imgSrc: "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/doc/img/fashion-mnist-sprite.png",
    },
    {
      value: "cifar10",
      label: "CIFAR-10",
      imgSrc: "https://www.cs.toronto.edu/~kriz/cifar10_sample.png",
    },
    {
      value: "cifar100",
      label: "CIFAR-100",
      imgSrc: "https://www.cs.toronto.edu/~kriz/cifar10_sample.png", // Replace with actual CIFAR-100 image
    },
  ];

  const handleLayerChange = (index: number, key: string, value: string) => {
    const newArch = [...arch];
    newArch[index][key] = value;
    setArch(newArch);
  };

  const addLayer = () => {
    setArch([...arch, { type: "", size: "" }]);
  };

  const removeLayer = (index: number) => {
    const newArch = arch.filter((_, i) => i !== index);
    setArch(newArch);
  };

  const sendIt = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const response = await fetch("http://127.0.0.1:1000/train", {
        method: "POST", // Ensure this is POST
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ dataset, arch, epochs }),
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      console.log(data);
    } catch (error) {
      console.error("Failed to fetch:", error.message);
    }
  };
  

  return (
    <div className="flex h-screen items-center justify-center">
      <form className="rounded-lg bg-white p-8 shadow-lg" onSubmit={sendIt}>
        <div className="m-5 flex items-center justify-center">
          <h1 className="text-2xl">Train Your Model Here</h1>
        </div>
        <div className="w-full border-2 border-solid border-black mb-4"></div>

        <div className="mt-4">
          <label className="flex items-center justify-center text-xl">
            Dataset
          </label>
          <div className="flex space-x-4 mt-4">
            {datasets.map((data) => (
              <label
                key={data.value}
                className="relative cursor-pointer"
              >
                <input
                  className="hidden"
                  name="dataset"
                  type="radio"
                  value={data.value}
                  checked={dataset === data.value}
                  onChange={(e) => setDataset(e.target.value)}
                />
                <img
                  src={data.imgSrc}
                  alt={data.label}
                  className={`h-20 w-20 rounded ${dataset === data.value ? "border-4 border-blue-500" : ""}`}
                />
                <span className="absolute inset-0 flex items-center justify-center text-white font-bold">
                  {data.label}
                </span>
              </label>
            ))}
          </div>
        </div>

        <div className="mt-4">
          <label className="flex items-center justify-center text-xl">
            Architecture
          </label>
          {arch.map((layer, index) => (
            <div key={index} className="mt-2 flex items-center space-x-4">
              <select
                className="w-full rounded-sm bg-[#f0f0f0] p-2"
                onChange={(e) => handleLayerChange(index, "type", e.target.value)}
                value={layer.type}
              >
                <option value="">Select Layer Type</option>
                <option value="linear">Linear</option>
                <option value="relu">ReLU</option>
                <option value="sigmoid">Sigmoid</option>
                <option value="batchnorm1d">BatchNorm1d</option>
                <option value="dropout">Dropout 20%</option>
                <option value="flatten">Flatten</option>
                {/* <option value="softmax">Softmax</option> */}
              </select>
              {(layer.type === "linear") && (
                <input
                  type="number"
                  placeholder="Size"
                  className="w-20 rounded-sm bg-[#f0f0f0] p-2"
                  onChange={(e) => handleLayerChange(index, "size", e.target.value)}
                  value={layer.size}
                />
              )}
              <button
                type="button"
                className="text-red-500"
                onClick={() => removeLayer(index)}
              >
                Remove
              </button>
            </div>
          ))}
          <button
            type="button"
            className="mt-2 w-full rounded-sm bg-blue-500 p-2 text-white"
            onClick={addLayer}
          >
            Add Layer
          </button>
        </div>

        <div className="mt-4">
          <label className="flex items-center justify-center text-xl">
            Epochs
          </label>
          <input
            type="number"
            className="w-full rounded-sm bg-[#f0f0f0] p-2"
            value={epochs}
            onChange={(e) => setEpochs(Number(e.target.value))}
          />
        </div>

        <div className="mt-4 flex justify-center">
          <button
            type="submit"
            className="rounded-sm bg-green-500 p-2 text-white"
          >
            Train Model
          </button>
        </div>
      </form>
    </div>
  );
};

export default Form;
