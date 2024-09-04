import { useState } from 'react';
import './App.css'
import Form from './components/form';
import Header from './components/header';

function App() {
  return (
    <div className='bg-1f1f1f'>
      <Header/>
      {/* <h1>golbok</h1> */}
      <main>
        <Form/>
      </main>
      
    </div>
  )
}
// https://storage.googleapis.com/tfds-data/visualization/fig/mnist-3.0.1.png

export default App
