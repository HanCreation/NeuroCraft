import React, { useState } from 'react';

interface FormData {
    dataset: string;
    arch: { type: string; size: string }[];
    epochs: number;
}

const Form: React.FC = () => {
    const [formData, setFormData] = useState<FormData>({
        dataset: '',
        arch: [],
        epochs: 0,
    });
    const [trainAccuracy, setTrainAccuracy] = useState<number | null>(null);
    const [testAccuracy, setTestAccuracy] = useState<number | null>(null);

    const handleSubmit = (event: React.FormEvent) => {
        event.preventDefault();
        fetch('http://127.0.0.1:1000/train', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formData),
        })
            .then((response) => response.json())
            .then((data) => {
                setTrainAccuracy(data.train_accuracy);
                setTestAccuracy(data.test_accuracy);
            })
            .catch((error) => {
                console.error(error);
            });
    };

    const handleChange = (event: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
        const { name, value } = event.target;
        setFormData((prevFormData) => ({
            ...prevFormData,
            [name]: value,
        }));
    };

    const handleArchChange = (event: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
        const { name, value } = event.target;
        const index = Number(name.split('-')[1]);
        setFormData((prevFormData) => {
            const updatedArch = [...prevFormData.arch];
            updatedArch[index] = { ...updatedArch[index], [name.includes('type') ? 'type' : 'size']: value };
            return {
                ...prevFormData,
                arch: updatedArch,
            };
        });
    };

    const handleAddLayer = () => {
        setFormData((prevFormData) => ({
            ...prevFormData,
            arch: [...prevFormData.arch, { type: '', size: '' }],
        }));
    };

    const handleRemoveLayer = (index: number) => {
        setFormData((prevFormData) => {
            const updatedArch = [...prevFormData.arch];
            updatedArch.splice(index, 1);
            return {
                ...prevFormData,
                arch: updatedArch,
            };
        });
    };

    return (
        <div className="flex justify-center items-center min-h-screen bg-gray-900">
            <form onSubmit={handleSubmit} className="w-full max-w-lg p-8 bg-white rounded-lg shadow-md">
                <h2 className="text-2xl font-bold mb-6 text-center">Training Configuration</h2>
                
                <div className="mb-4">
                    <label htmlFor="dataset" className="block text-gray-700 font-bold mb-2">Dataset:</label>
                    <select
                        name="dataset"
                        value={formData.dataset}
                        onChange={handleChange}
                        className="w-full p-2 border border-gray-300 rounded"
                    >
                        <option value="">Select a dataset</option>
                        <option value="mnist">MNIST</option>
                        <option value="fashion_mnist">Fashion MNIST</option>
                        <option value="cifar10">CIFAR-10</option>
                    </select>
                </div>

                <div className="mb-4">
                    <label htmlFor="arch" className="block text-gray-700 font-bold mb-2">Architecture:</label>
                    {formData.arch.map((layer, index) => (
                        <div key={index} className="flex items-center mb-2">
                            <select
                                name={`arch-${index}-type`}
                                value={layer.type}
                                onChange={handleArchChange}
                                className="w-1/2 p-2 border border-gray-300 rounded mr-2"
                            >
                                <option value="">Select a layer type</option>
                                <option value="linear">Linear / Dense</option>
                                <option value="relu">ReLU</option>
                                <option value="sigmoid">Sigmoid</option>
                                <option value="batchnorm1d">Batch Normalization</option>
                                <option value="dropout">Dropout</option>
                                <option value="flatten">Flatten</option>
                                <option value="softmax">Softmax</option>
                            </select>
                            {layer.type === 'linear' && (
                                <input
                                    type="text"
                                    name={`arch-${index}-size`}
                                    value={layer.size}
                                    onChange={handleArchChange}
                                    placeholder="Layer size"
                                    className="w-1/2 p-2 border border-gray-300 rounded mr-2"
                                />
                            )}
                            <button
                                type="button"
                                onClick={() => handleRemoveLayer(index)}
                                className="text-red-500 hover:text-red-700"
                            >
                                Remove
                            </button>
                        </div>
                    ))}
                    <button
                        type="button"
                        onClick={handleAddLayer}
                        className="w-full p-2 bg-blue-500 text-white rounded hover:bg-blue-700"
                    >
                        Add Layer
                    </button>
                </div>

                <div className="mb-4">
                    <label htmlFor="epochs" className="block text-gray-700 font-bold mb-2">Epochs:</label>
                    <input
                        type="number"
                        name="epochs"
                        value={formData.epochs}
                        onChange={handleChange}
                        className="w-full p-2 border border-gray-300 rounded"
                    />
                </div>

                <button
                    type="submit"
                    className="w-full p-2 bg-green-500 text-white rounded hover:bg-green-700"
                >
                    Submit
                </button>

                {trainAccuracy !== null && testAccuracy !== null && (
                    <div className="mt-4">
                        <h3 className="text-lg font-bold mb-2">Training Results:</h3>
                        <p>Train Accuracy: {trainAccuracy}</p>
                        <p>Test Accuracy: {testAccuracy}</p>
                    </div>
                )}
            </form>
        </div>
    );
};

export default Form;
