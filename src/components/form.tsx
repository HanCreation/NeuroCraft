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

  //Init awal semuanya null, nanti bakal berubah seterah response JSON
  const [trainAccuracy, setTrainAccuracy] = useState<number | null>(null);
  const [testAccuracy, setTestAccuracy] = useState<number | null>(null);

  //React.FormEvent type event buat menangani form, input, textarea, select
  const handleSubmit = (event: React.FormEvent) => {
    event.preventDefault();
    //Call the API
    fetch("http://127.0.0.1:1000/train", {
      method: "POST", //Method call
      headers: {
        //Header HTTP yang dikirmkan, application/json berarti tipe datanya JSON
        "Content-Type": "application/json",
      },
      //Body dari HTTPnya, object formData di convert jadi string JSON
      body: JSON.stringify(formData),
    })
      //Take HTTP response
      .then((response) => response.json())
      //Ambil object JSON, ambil data dari JSON, terus set state datanya
      .then((data) => {
        setTrainAccuracy(data.train_accuracy);
        setTestAccuracy(data.test_accuracy);
      })
      //from python -> jsonify({'train_accuracy': train_accuracy, 'test_accuracy': test_accuracy})

      //Nangkep error
      .catch((error) => {
        console.error(error);
      });
  };

  const handleArchChange = (
    event: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>,
    //DIpake buat menangani perubahan pada element form
  ) => {
    const { name, value } = event.target;
    const index = Number(name.split("-")[1]); //{`arch-${idx}-type`}

    setFormData((prevFormData) => {
      const updatedArch = [...prevFormData.arch]; //Copy the data
      updatedArch[index] = {
        ...updatedArch[index],
        [name.includes("type") ? "type" : "size"]: value,
        //Namenya type atau bukan, kalau bukan set jadi size -> "size" : value buat JSON
      };
      return {
        ...prevFormData,
        arch: updatedArch,
      };
    });
  };

  return (
    //Div is centered flex+justify-center+items-center+min-h-screen
    <div className="flex min-h-screen items-center justify-center">
      <form
        className="w-full max-w-xl rounded-lg bg-white p-8 shadow-md"
        onSubmit={handleSubmit}
      >
        <h1 className="mb-6 text-center text-4xl font-bold">
          Neural Network Configuration
        </h1>

        <div className="m-7">
          <label className="flex justify-center text-center text-2xl">Dataset</label>
          <div className="mt-4 flex space-x-4 justify-center"> 
            {/* // Render a div element with the specified class names and flexbox properties */}
            {datasets.map((data) => ( // Iterate over the 'datasets' array and render the following elements for each object in the array
              <label key={data.value} className="relative cursor-pointer"> 
              {/* // Render a label element with the specified class names and key attribute */}
                <input
                  className="hidden"
                  name="dataset"
                  type="radio"
                  value={data.value}
                  checked={formData.dataset === data.value}
                  onChange={(e) => setFormData({...formData, dataset: e.target.value})}
                /> 
                {/* // Render an input element with the specified attributes and event handler */}
                <img
                  src={data.imageSrc}
                  alt={data.label}
                  className={`h-20 w-20 rounded ${formData.dataset === data.value ? "border-4 border-blue-500" : ""}`} // Render an img element with the specified attributes and conditional class name
                />
                <span className="absolute inset-0 flex items-center justify-center font-bold text-1f1f1f mt-24 text-center"> 
                  {/* // Render a span element with the specified class names and text content */}
                  {data.label}
                </span>
              </label>
            ))}
          </div>
        </div>
        <div className="mb-4">
          <label htmlFor="arch" className="mb-2 flex items-center">
            Architechture
          </label>
          {/* Iterate over formData arch, need 2 parameter since to keep track of the structure of the layer */}
          {formData.arch.map((layer, idx) => (
            <div key={idx} className="mb-2 flex items-center">
              <select
                name={`arch-${idx}-type`}
                value={layer.type}
                onChange={handleArchChange}
                className="w-full rounded border border-gray-300 p-2"
              >
              
              </select>
            </div>
          ))}
        </div>
      </form>
    </div>
  );
};

export default Form;
