import React, { useState } from 'react'


const form: React.FC = () => {
    return(
        <div className="flex items-center justify-center h-screen">
            <form className="p-8 bg-white shadow-lg">
             <div className=''>
             <fieldset>
            <legend className="sr-only">Countries</legend>

            <label htmlFor='username' className='block text-gray-700  text-center font-bold mb-2'>Dataset</label>
            <div className="flex items-center justify-center mb-4">
                <input id="country-option-1" type="radio" name="countries" value="USA" className="w-4 h-4 border-gray-300 focus:ring-2 focus:ring-blue-300 dark:focus:ring-blue-600 dark:focus:bg-blue-600 dark:bg-gray-700 dark:border-gray-600" checked></input>
                <label id="country-option-1" className="block ms-2  text-sm font-medium text-1f1f1f">
                MNIST
                </label>
            </div>
            <div className="flex items-center justify-center">
            <input id="country-option-1" type="radio" name="countries" value="USA" className="w-4 h-4 border-gray-300 focus:ring-2 focus:ring-blue-300 dark:focus:ring-blue-600 dark:focus:bg-blue-600 dark:bg-gray-700 dark:border-gray-600"></input>
                <label id="country-option-1" className="block ms-2  text-sm font-medium text-1f1f1f">
                Fashion-MNIST
                </label>
            </div>
            </fieldset>
               
             </div>
            </form>
           
        </div>
    )
}

export default form;