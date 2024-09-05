import React, { useState } from "react";

//JSON DATA
// {
//     "dataset": "mnist",
//     "arch": [
//         { "type": "linear", "size": "128" },
//         { "type": "relu", "size": "" },
//         { "type": "linear", "size": "64" },
//         { "type": "softmax", "size": "" }
//     ],
//     "epochs": 10
// }
//Ensure form data will have the data template
interface FormData {
  dataset: string;
  arch: { type: string; size: string }[]; //array of these
  epochs: number;
}

const Form: React.FC = () => {
  //useState mengelola state di dalam FormData -> Generic Type
  //komponen fungsional adalah fungsi yang return something ke front endnya
  const [formData, setFormData] = useState<FormData>({
    dataset: "",
    arch: [],
    epochs: 0,
  });

  const datasets = [
    {
      value: "mnist",
      label: "MNIST",
      imageSrc:
        "https://storage.googleapis.com/tfds-data/visualization/fig/mnist-3.0.1.png",
    },
    {
      value: "fashion_mnist",
      label: "Fashion MNIST",
      imageSrc:
        "https://storage.googleapis.com/tfds-data/visualization/fig/fashion_mnist-3.0.1.png",
    },
  ];

  //Init awal semuanya null
  const [trainAccuracy, setTrainAccuracy] = useState<number | null>(null);
  const [testAccuracy, setTestAccuracy] = useState<number | null>(null);

  //React.formEvent type event buat menangani form, input, textarea, select
  const handleSubmit = (event: React.FormEvent) => {
    event.preventDefault();
  };

  return (
    //Div is centered flex+justify-center+items-center+min-h-screen
    <div className="flex min-h-screen items-center justify-center">
      <form
        className="w-full max-w-xl rounded-lg bg-white p-8 shadow-md"
        onSubmit={handleSubmit}
      >
        <h1 className="mb-6 text-center text-2xl font-bold">
          Neural Network Configuration
        </h1>

        <div className="">
          <label className="flex justify-center text-center">Dataset</label>
          <div>
            {datasets.map((data) => (
              <label
                className="relative cursor-pointer"
                key={data.value}
              >
                <input className="hidden" type="radio" name="dataset" value={data.value} checked={formData.dataset === data.value}/>

              </label>
            ))}
          </div>
        </div>
      </form>
    </div>
  );
};

export default Form;
