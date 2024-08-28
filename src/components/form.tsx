import React, { useState } from 'react'


const form: React.FC = () => {
    return(
        <>

           <form className='max-w-sm mx-auto align-cente'>
             <div className='mb-5 items-center'>
                <label htmlFor='username' className='block text-gray-700 text-sm font-bold mb-2'>Username</label>
                <input type="email" id="email" className="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block w-full p-2.5 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500" placeholder="name@flowbite.com" required />
             </div>

            </form>
        </>

    )
}

export default form;