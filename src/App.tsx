import { useState } from 'react';
import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import './App.css'
import Form from './components/form';
import Header from './components/header';

function App() {
  return (
    <div className='bg-1f1f1f'>
      <Header/>
      <Form/>
    </div>
  )
}
// https://storage.googleapis.com/tfds-data/visualization/fig/mnist-3.0.1.png

export default App
