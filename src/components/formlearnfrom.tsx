import React, { useState } from "react"; // Import React and useState from the react package

const Form: React.FC = () => { // Define a functional component called Form
  const [dataset, setDataset] = useState("mnist"); // Declare a state variable 'dataset' and a function to update it 'setDataset', initialized with the value "mnist"
  const [arch, setArch] = useState([{ type: "", size: "" }]); // Declare a state variable 'arch' and a function to update it 'setArch', initialized with an array containing an object with empty 'type' and 'size' properties
  const [epochs, setEpochs] = useState(10); // Declare a state variable 'epochs' and a function to update it 'setEpochs', initialized with the value 10

  const datasets = [ // Declare an array of objects representing different datasets
    {
      value: "mnist",
      label: "MNIST",
      imgSrc:
        "https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png",
    },
    {
      value: "fashion_mnist",
      label: "Fashion MNIST",
      imgSrc:
        "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/doc/img/fashion-mnist-sprite.png",
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

  const handleLayerChange = (index: number, key: string, value: string) => { // Define a function 'handleLayerChange' that takes an index, key, and value as parameters
    const newArch = [...arch]; // Create a copy of the 'arch' array using the spread operator
    newArch[index][key] = value; // Update the value at the specified index and key in the 'newArch' array
    setArch(newArch); // Update the 'arch' state variable with the updated 'newArch' array
  };

  const addLayer = () => { // Define a function 'addLayer'
    setArch([...arch, { type: "", size: "" }]); // Add a new object with empty 'type' and 'size' properties to the 'arch' array and update the 'arch' state variable
  };

  const removeLayer = (index: number) => { // Define a function 'removeLayer' that takes an index as a parameter
    const newArch = arch.filter((_, i) => i !== index); // Create a new array 'newArch' by filtering out the element at the specified index from the 'arch' array
    setArch(newArch); // Update the 'arch' state variable with the updated 'newArch' array
  };

  const sendIt = async (e: React.FormEvent) => { // Define an asynchronous function 'sendIt' that takes a React form event as a parameter
    e.preventDefault(); // Prevent the default form submission behavior
    try {
      const response = await fetch("http://127.0.0.1:1000/train", { // Send a POST request to the specified URL
        method: "POST", // Set the request method to POST
        headers: {
          "Content-Type": "application/json", // Set the request header to specify the content type as JSON
        },
        body: JSON.stringify({ dataset, arch, epochs }), // Convert the data to JSON format and include it in the request body
      });

      if (!response.ok) { // If the response status is not ok
        throw new Error(`HTTP error! status: ${response.status}`); // Throw an error with the response status
      }

      const data = await response.json(); // Parse the response data as JSON
      console.log(data); // Log the data to the console
    } catch (error) {
      console.error("Failed to fetch:", error.message); // Log the error message to the console
    }
  };

  return (
    <div className="flex h-screen items-center justify-center">
       {/* // Render a div element with the specified class names and flexbox properties */}
      <form className="rounded-lg bg-white p-8 shadow-lg" onSubmit={sendIt}>
         {/* // Render a form element with the specified class names and submit event handler */}
        <div className="m-5 flex items-center justify-center"> 
          {/* // Render a div element with the specified class names and flexbox properties */}
          <h1 className="text-2xl">Train Your Model Here</h1> 
          {/* / Render an h1 element with the specified class name and text content */}
        </div>
        <div className="mb-4 w-full border-2 border-solid border-black"></div> 
        {/* // Render a div element with the specified class names and border properties */}

        <div className="mt-4">
          <label className="flex items-center justify-center text-xl"> 
            {/* // Render a label element with the specified class names and text content */}
            Dataset
          </label>
          <div className="mt-4 flex space-x-4"> 
            {/* // Render a div element with the specified class names and flexbox properties */}
            {datasets.map((data) => ( // Iterate over the 'datasets' array and render the following elements for each object in the array
              <label key={data.value} className="relative cursor-pointer"> 
              {/* // Render a label element with the specified class names and key attribute */}
                <input
                  className="hidden"
                  name="dataset"
                  type="radio"
                  value={data.value}
                  checked={dataset === data.value}
                  onChange={(e) => setDataset(e.target.value)}
                /> 
                {/* // Render an input element with the specified attributes and event handler */}
                <img
                  src={data.imgSrc}
                  alt={data.label}
                  className={`h-20 w-20 rounded ${dataset === data.value ? "border-4 border-blue-500" : ""}`} // Render an img element with the specified attributes and conditional class name
                />
                <span className="absolute inset-0 flex items-center justify-center font-bold text-white"> 
                  {/* // Render a span element with the specified class names and text content */}
                  {data.label}
                </span>
              </label>
            ))}
          </div>
        </div>

        <div className="mt-4">
          <label className="flex items-center justify-center text-xl"> 
            {/* // Render a label element with the specified class names and text content */}
            Architecture
          </label>
          {arch.map((layer, index) => ( 
            // Iterate over the 'arch' array and render the following elements for each object in the array
            <div key={index} className="mt-2 flex items-center space-x-4"> 
            {/* // Render a div element with the specified class names and flexbox properties */}
              <select
                className="w-full rounded-sm bg-[#f0f0f0] p-2"
                onChange={(e) =>
                  handleLayerChange(index, "type", e.target.value)
                }
                value={layer.type}
              > 
              {/* // Render a select element with the specified attributes and event handler */}
                <option value="">Select Layer Type</option> 
                {/* // Render an option element with the specified value and text content */}
                <option value="linear">Linear</option> 
                {/* // Render an option element with the specified value and text content */}
                <option value="relu">ReLU</option> 
                {/* // Render an option element with the specified value and text content */}
                <option value="sigmoid">Sigmoid</option> 
                {/* // Render an option element with the specified value and text content */}
                <option value="batchnorm1d">BatchNorm1d</option> 
                {/* // Render an option element with the specified value and text content */}
                <option value="dropout">Dropout 20%</option> 
                {/* // Render an option element with the specified value and text content */}
                <option value="flatten">Flatten</option> 
                {/* // Render an option element with the specified value and text content */}
                {/* <option value="softmax">Softmax</option> */} 
                {/* // Render a commented out option element */}
              </select>
              {layer.type === "linear" && ( // Render the following input element if the 'layer.type' is equal to "linear"
                <input
                  type="number"
                  placeholder="Size"
                  className="w-20 rounded-sm bg-[#f0f0f0] p-2"
                  onChange={(e) =>
                    handleLayerChange(index, "size", e.target.value)
                  }
                  value={layer.size}
                />
              )}
              <button
                type="button"
                className="text-red-500"
                onClick={() => removeLayer(index)}
              > 
              {/* // Render a button element with the specified attributes and event handler */}
                Remove
              </button>
            </div>
          ))}
          <button
            type="button"
            className="mt-2 w-full rounded-sm bg-blue-500 p-2 text-white"
            onClick={addLayer}
          > 
          {/* // Render a button element with the specified attributes and event handler */}
            Add Layer
          </button>
        </div>

        <div className="mt-4">
          <label className="flex items-center justify-center text-xl"> 
            {/* // Render a label element with the specified class names and text content */}
            Epochs
          </label>
          <input
            type="number"
            className="w-full rounded-sm bg-[#f0f0f0] p-2"
            value={epochs}
            onChange={(e) => setEpochs(Number(e.target.value))}
          /> 
          {/* // Render an input element with the specified attributes and event handler */}
        </div>

        <div className="mt-4 flex justify-center">
          <button
            type="submit"
            className="rounded-sm bg-green-500 p-2 text-white"
          > 
          {/* // Render a button element with the specified attributes */}
            Train Model
          </button>
        </div>
      </form>
    </div>
  );
};

export default Form; // Export the Form component as the default export
