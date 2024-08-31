import React, { useState } from "react";


const form: React.FC = () => {
  const [dataset, setDataset] = useState("mnist");


  return (
    <div className="flex h-screen items-center justify-center">
      <form className="rounded-lg bg-white p-8 shadow-lg">
        <div className="m-5 flex items-center justify-center">
          <h1 className="text-2xl">Train Your Model Here</h1>
        </div>
        <div className="w-5000000 border-2 border-solid border-black"></div>
        <div className="mt-4">
          <label className="flex items-center justify-center text-xl">
            Dataset
          </label>
        </div>

        <div className="flex space-x-2 text-sm font-light text-black">
          <div>
            <label className="relative">
              <input
                className="flex h-9 w-10 appearance-none items-center justify-center rounded-sm bg-[#1f1f1f]"
                name="size"
                type="radio"
                value="mnist"
                checked
              ></input>
              {/* <div className="absolute"> */}
              <img
                  src="https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png"
                  alt="mnist"
                  className="h-20 w-20"
                ></img>
              {/* </div> */}
              <span className="absolute inset-0 flex items-center justify-center text-white">
                MNIST
              </span>
            </label>
            <div>
              
            </div>
          </div>
        </div>
      </form>
    </div>
  );
};

export default form;
