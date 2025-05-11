import React, { useState } from 'react';
import { ArrowRight } from 'lucide-react';

// Main flowchart component
export default function ImageClassificationFlowchart() {
  const [hoveredNode, setHoveredNode] = useState(null);
  
  // Node definitions with data
  const nodes = {
    start: {
      id: 'start',
      title: 'Start',
      description: '1,200 Original Images',
      icon: '📁',
      color: 'bg-blue-100 border-blue-500'
    },
    preprocessing: {
      id: 'preprocessing',
      title: 'Preprocessing',
      description: 'Clean and prepare image data',
      icon: '🔧',
      color: 'bg-indigo-100 border-indigo-500'
    },
    augmentation: {
      id: 'augmentation',
      title: 'Augmentation',
      description: 'Using Albumentations library',
      icon: '🔄',
      color: 'bg-purple-100 border-purple-500'
    },
    training: {
      id: 'training',
      title: 'Model Training',
      description: 'Train multiple model architectures',
      icon: '⚙️',
      color: 'bg-amber-100 border-amber-500'
    },
    model1: {
      id: 'model1',
      title: 'Model 1',
      description: 'LightCNN (with augmentation)',
      icon: '📊',
      color: 'bg-green-100 border-green-500'
    },
    model2: {
      id: 'model2',
      title: 'Model 2',
      description: 'LightCNN (no augmentation)',
      icon: '📊',
      color: 'bg-green-100 border-green-500'
    },
    model3: {
      id: 'model3',
      title: 'Model 3',
      description: 'MobileNetV2',
      icon: '📊',
      color: 'bg-green-100 border-green-500'
    },
    eval1: {
      id: 'eval1',
      title: 'Evaluation 1',
      description: 'Accuracy metrics for Model 1',
      icon: '📈',
      color: 'bg-teal-100 border-teal-500'
    },
    eval2: {
      id: 'eval2',
      title: 'Evaluation 2',
      description: 'Accuracy metrics for Model 2',
      icon: '📈',
      color: 'bg-teal-100 border-teal-500'
    },
    eval3: {
      id: 'eval3',
      title: 'Evaluation 3',
      description: 'Accuracy metrics for Model 3',
      icon: '📈',
      color: 'bg-teal-100 border-teal-500'
    },
    selection: {
      id: 'selection',
      title: 'Model Selection',
      description: 'Based on performance metrics',
      icon: '🏆',
      color: 'bg-orange-100 border-orange-500'
    },
    final: {
      id: 'final',
      title: 'Final System',
      description: 'Deployed classification system',
      icon: '🚀',
      color: 'bg-red-100 border-red-500'
    }
  };

  // Define connections between nodes
  const connections = [
    { from: 'start', to: 'preprocessing' },
    { from: 'preprocessing', to: 'augmentation' },
    { from: 'augmentation', to: 'training' },
    { from: 'training', to: 'model1' },
    { from: 'training', to: 'model2' },
    { from: 'training', to: 'model3' },
    { from: 'model1', to: 'eval1' },
    { from: 'model2', to: 'eval2' },
    { from: 'model3', to: 'eval3' },
    { from: 'eval1', to: 'selection' },
    { from: 'eval2', to: 'selection' },
    { from: 'eval3', to: 'selection' },
    { from: 'selection', to: 'final' }
  ];

  // Node component with hover effects
  const FlowNode = ({ node }) => {
    return (
      <div 
        className={`p-3 rounded-lg shadow-md border-l-4 transition-all ${node.color} ${hoveredNode === node.id ? 'scale-105 shadow-lg' : ''}`}
        style={{ width: '220px' }}
        onMouseEnter={() => setHoveredNode(node.id)}
        onMouseLeave={() => setHoveredNode(null)}
      >
        <div className="flex items-center mb-2">
          <span className="text-2xl mr-2">{node.icon}</span>
          <h3 className="font-bold text-gray-800">{node.title}</h3>
        </div>
        <p className="text-sm text-gray-600">{node.description}</p>
      </div>
    );
  };

  // Connection arrow with animation on hover
  const Connection = ({ from, to, isVertical = false }) => {
    const isHighlighted = hoveredNode === from || hoveredNode === to;
    
    return (
      <div className={`flex items-center justify-center ${isVertical ? 'flex-col h-8' : 'w-6'}`}>
        <div className={`flex items-center ${isVertical ? 'h-full' : 'w-full'}`}>
          <div className={`${isVertical ? 'w-0.5 h-full' : 'w-full h-0.5'} bg-gray-400 ${isHighlighted ? 'bg-blue-500' : ''}`}></div>
        </div>
        <ArrowRight 
          size={16} 
          className={`${isVertical ? 'rotate-90' : ''} ${isHighlighted ? 'text-blue-500' : 'text-gray-400'}`} 
        />
      </div>
    );
  };

  return (
    <div className="p-8 bg-white rounded-xl shadow-lg max-w-5xl mx-auto">
      <h2 className="text-2xl font-bold text-center mb-8 text-gray-800">Image Classification Workflow</h2>
      
      {/* Row 1 */}
      <div className="flex justify-center items-center mb-6">
        <FlowNode node={nodes.start} />
        <Connection from="start" to="preprocessing" />
        <FlowNode node={nodes.preprocessing} />
        <Connection from="preprocessing" to="augmentation" />
        <FlowNode node={nodes.augmentation} />
      </div>
      
      {/* Vertical connection */}
      <div className="flex justify-center mb-6">
        <Connection from="augmentation" to="training" isVertical={true} />
      </div>
      
      {/* Row 2 */}
      <div className="flex justify-center items-center mb-6">
        <FlowNode node={nodes.training} />
      </div>
      
      {/* Row 3 */}
      <div className="flex justify-center items-center mb-6">
        <FlowNode node={nodes.model1} />
        <div className="w-6"></div>
        <FlowNode node={nodes.model2} />
        <div className="w-6"></div>
        <FlowNode node={nodes.model3} />
      </div>
      
      {/* Row 4 */}
      <div className="flex justify-center items-center mb-6">
        <Connection from="model1" to="eval1" isVertical={true} />
        <div className="w-6"></div>
        <Connection from="model2" to="eval2" isVertical={true} />
        <div className="w-6"></div>
        <Connection from="model3" to="eval3" isVertical={true} />
      </div>
      
      {/* Row 5 */}
      <div className="flex justify-center items-center mb-6">
        <FlowNode node={nodes.eval1} />
        <div className="w-6"></div>
        <FlowNode node={nodes.eval2} />
        <div className="w-6"></div>
        <FlowNode node={nodes.eval3} />
      </div>
      
      {/* Row 6 - converging connections */}
      <div className="flex justify-center items-center mb-6">
        <div className="flex flex-col items-center">
          <Connection from="eval1" to="selection" isVertical={true} />
        </div>
        <div className="w-6"></div>
        <div className="flex flex-col items-center">
          <Connection from="eval2" to="selection" isVertical={true} />
        </div>
        <div className="w-6"></div>
        <div className="flex flex-col items-center">
          <Connection from="eval3" to="selection" isVertical={true} />
        </div>
      </div>
      
      {/* Row 7 */}
      <div className="flex justify-center items-center mb-6">
        <FlowNode node={nodes.selection} />
      </div>
      
      {/* Row 8 */}
      <div className="flex justify-center items-center mb-6">
        <Connection from="selection" to="final" isVertical={true} />
      </div>
      
      {/* Row 9 */}
      <div className="flex justify-center items-center">
        <FlowNode node={nodes.final} />
      </div>
      
      <div className="mt-8 text-center text-sm text-gray-500">
        <p>Hover over each component to highlight connections</p>
      </div>
    </div>
  );
}