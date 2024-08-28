import React, { useState } from 'react'


const form: React.FC = () => {
    return(
        <div className="flex items-center justify-center h-screen">
           <form className="p-8 bg-white shadow-lg">
             <div className=''>
                <label htmlFor='username' className='block text-gray-700 text-sm font-bold mb-2'>Username</label>
             </div>
            </form>
        </div>
    )
}

export default form;