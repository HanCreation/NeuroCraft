import React, { useState } from "react";
import { ClipLoader } from "react-spinners";

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
    dataset: "mnist",
    arch: [{ type: "", size: "" }],
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

  //Loading
  const [loading, setLoading] = useState<boolean>(false);
  const [isButtonClicked, setIsButtonClicked] = useState<boolean>(false);

  //React.FormEvent type event buat menangani form, input, textarea, select
  const handleSubmit = (event: React.FormEvent) => {
    event.preventDefault();
    //Enable loading state
    setLoading(true);
    setIsButtonClicked(true);

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
      })
      //Setelah selesai loading, set loading jadi false
      .finally(() => {
        setLoading(false);
        setIsButtonClicked(false);
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

  // Same as like:
  // const handleLayerChange = (index: number, key: string, value: string) => { ...the syntax...}
  // key -> using keyof which either type or size according to the interface itself (formData arch [0] aka the left part)
  // Specifically:
  // FormData['arch'][0] is equivalent to { type: string; size: string }.
  // It’s like saying, "Give me the type of the first element in this array," which tells TypeScript what each element (object) in the arch array looks like.
  function handleSize(
    index: number,
    key: keyof FormData["arch"][0],
    value: string,
  ) {
    //Copy of arch
    const newArch = [...formData.arch];
    //set newArch with the new value
    newArch[index][key] = value;
    //Create new formData with new value set
    setFormData({ ...formData, arch: newArch });
  }

  function removeLayer(index: number) {
    // Create a new array 'newArch' by filtering out the element at the specified index from the 'arch' array (i!=index)
    const newArch = formData.arch.filter((_, i) => i !== index);
    setFormData({ ...formData, arch: newArch });
  }

  function addLayer() {
    //Memperbaharui state formdata dengan cara membuat salinan form data kemudian tambahin arch kosong yg bakal diedit nanti (Alias kita nambahin archnya aja)
    setFormData({
      ...formData,
      arch: [...formData.arch, { type: "", size: "" }],
    });
  }

  return (
    //Div is centered flex+justify-center+items-center+min-h-screen
    <div className="flex min-h-screen items-center justify-center">
      <form
        className="w-full max-w-xl rounded-lg bg-white p-8 shadow-md"
        onSubmit={handleSubmit}
      >
        <h1 className="mb-6 text-center text-3xl font-bold">
          Neural Network Configuration
        </h1>

        <div className="m-7 mb-20">
          <label className="flex justify-center text-center text-xl">
            Dataset
          </label>
          <div className="mt-4 flex justify-center space-x-4">
            {/* // Render a div element with the specified class names and flexbox properties */}
            {datasets.map(
              (
                data, // Iterate over the 'datasets' array and render the following elements for each object in the array
              ) => (
                <label key={data.value} className="relative cursor-pointer">
                  {/* // Render a label element with the specified class names and key attribute */}
                  <input
                    className="hidden"
                    name="dataset"
                    type="radio"
                    value={data.value}
                    checked={formData.dataset === data.value}
                    // onChange secara real time dipanggil saat elemen input berubah, kemudian dilempar ke fungsi setFormData yang menerima paremeter e (sebuah object event). terus dia bakal buat formData baru dengan dataset udah di update jadi e.target.value (target, elemen input yg memicu event, valuenya itu value dari elemennya)
                    onChange={(e) =>
                      setFormData({ ...formData, dataset: e.target.value })
                    }
                  />
                  {/* // Render an input element with the specified attributes and event handler */}
                  <img
                    src={data.imageSrc}
                    alt={data.label}
                    className={`h-30 w-30 rounded ${formData.dataset === data.value ? "border-4 border-blue-500" : ""}`} // Render an img element with the specified attributes and conditional class name
                  />
                  <span className="inset-0 flex items-center justify-center text-center font-bold text-1f1f1f">
                    {/* // Render a span element with the specified class names and text content */}
                    {data.label}
                  </span>
                </label>
              ),
            )}
          </div>
        </div>
        <div className="mb-4">
          <label htmlFor="arch" className="mb-2 flex items-center text-xl">
            Neural Network Architechture
          </label>
          {/* Iterate over formData arch, need 2 parameter since to keep track of the structure of the layer */}
          {/* <h1>luar</h1> */}
          {formData.arch.map((layer, idx) => (
            <div key={idx} className="mb-2 flex items-center">
              {/* // Render a select element with the specified attributes and event handler */}
              <div className="flex w-full items-center">
                <select
                  name={`arch-${idx}-type`}
                  value={layer.type}
                  onChange={handleArchChange}
                  className="h-10 flex-grow rounded border border-gray-300 p-2"
                >
                  {/* // Render options element with the specified value and text content */}
                  <option value="">Select Layer Type</option>
                  {/* // Render an option element with the specified value and text content */}
                  <option value="linear">Linear</option>
                  <option value="relu">ReLU</option>
                  <option value="sigmoid">Sigmoid</option>
                  <option value="batchnorm1d">BatchNorm1d</option>
                  <option value="dropout">Dropout 20%</option>
                  {/* // Render an option element with the specified value and text content */}
                  {/* <option value="flatten">Flatten</option> */}
                  {/* // Render an option element with the specified value and text content */}
                  {/* <option value="softmax">Softmax</option> */}
                  {/* // Render a commented out option element */}
                </select>
                {/* if layer is linear berarti kita munculin input value.size */}
                {layer.type === "linear" && (
                  <input
                    type="number"
                    placeholder="Size"
                    className="w-20 rounded-sm bg-[#f0f0f0] p-2"
                    onChange={(e) => handleSize(idx, "size", e.target.value)}
                    value={layer.size}
                  />
                )}
                <button
                  type="button"
                  className="ml-2 h-10 text-red-500"
                  onClick={() => removeLayer(idx)}
                >
                  Remove Layer
                </button>
              </div>
            </div>
          ))}
          <button
            className="mt-2 w-full rounded-sm bg-blue-500 p-2 text-white"
            onClick={addLayer}
            type="button"
          >
            Add Layer
          </button>
        </div>
        <div className="mb-4">
          <label htmlFor="epochs" className="mb-2 block text-lg">
            Epochs:
          </label>
          <input
            type="number"
            name="epochs"
            className="w-full rounded-sm border-2 border-[#c2c2c2] bg-[#f0f0f0] p-2 text-black"
            value={formData.epochs}
            onChange={(e) =>
              setFormData({ ...formData, epochs: parseInt(e.target.value) })
            }
          />
        </div>
        <div className="flex justify-center">
          <button
            //If button sedang di click, maka warnanya gray dan tidak bisa dipencet
            // hover: untuk efek hover
            className={`relative rounded-sm p-2 ${isButtonClicked ? "bg-gray-500 text-gray-300" : "bg-green-500 text-white hover:bg-green-600"}`}
            type="submit"
            disabled={isButtonClicked}
          >
            Train Your Crafted Neurons
            <div className="absolute inset-0 flex items-center justify-center">
              {loading && (
                <ClipLoader color="#fff" size={15} loading={loading} />
              )}
            </div>
          </button>
        </div>

        {/* Render train and test accuracy when it is done */}
        {trainAccuracy !== null && testAccuracy !== null && (
          <div className="mt-4">
            <h3 className="mb-2 text-lg font-bold">
              {/* Ternary Operation if else of the dataset */}
              {formData.dataset === "mnist"
                ? "Training Results on MNIST Digit dataset"
                : "Training Results on Fashion MNIST dataset"}
            </h3>
            <p>Train Accuracy: {trainAccuracy}</p>
            <p>Test Accuracy: {testAccuracy}</p>
          </div>
        )}
      </form>
    </div>
  );
};

export default Form;
