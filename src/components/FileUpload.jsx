import React, { useRef, useState } from 'react';
import { Upload, Image, FileSpreadsheet } from 'lucide-react';

const FileUpload = ({ 
  type, 
  accept, 
  onUpload, 
  uploadedFile, 
  title, 
  description 
}) => {
  const fileInputRef = useRef(null);
  const [dragOver, setDragOver] = useState(false);
  const [imagePreview, setImagePreview] = useState(null);

  const handleFileSelect = (file) => {
    if (!file) return;

    // Validate file type
    const validTypes = accept.split(',').map(type => type.trim());
    const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
    
    if (!validTypes.includes(fileExtension)) {
      alert(`Please upload a valid file type: ${accept}`);
      return;
    }

    // Validate file size (50MB limit)
    if (file.size > 50 * 1024 * 1024) {
      alert('File is too large. Please use a file smaller than 50MB');
      return;
    }

    // Generate image preview for MRI files
    if (type === 'mri' && file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setImagePreview(e.target.result);
      };
      reader.readAsDataURL(file);
    }

    onUpload(file);
  };

  const handleFileInputChange = (e) => {
    const file = e.target.files[0];
    handleFileSelect(file);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    
    const file = e.dataTransfer.files[0];
    handleFileSelect(file);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setDragOver(false);
  };

  const openFileDialog = () => {
    fileInputRef.current?.click();
  };

  const getIcon = () => {
    if (type === 'mri') {
      return <Image size={32} className="upload-icon" />;
    } else {
      return <FileSpreadsheet size={32} className="upload-icon" />;
    }
  };

  const getFileInfo = () => {
    if (!uploadedFile) return null;
    
    const sizeInMB = (uploadedFile.size / 1024 / 1024).toFixed(2);
    return `${uploadedFile.name} (${sizeInMB} MB)`;
  };

  return (
    <div className={`upload-area ${dragOver ? 'dragover' : ''}`}>
      <h3>{title}</h3>
      <p>{description}</p>
      
      <div
        className="upload-zone"
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onClick={openFileDialog}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept={accept}
          onChange={handleFileInputChange}
          style={{ display: 'none' }}
        />
        
        {!uploadedFile ? (
          <div className="upload-placeholder">
            {getIcon()}
            <Upload size={24} className="upload-arrow" />
            <span>Click to browse or drag files here</span>
          </div>
        ) : (
          <div className="file-feedback">
            <div className="file-info">
              ✓ {type === 'mri' ? 'MRI' : 'Biomarker data'} uploaded successfully
              <br />
              <span className="file-details">{getFileInfo()}</span>
            </div>
            
            {imagePreview && type === 'mri' && (
              <div className="image-preview">
                <img 
                  src={imagePreview} 
                  alt="MRI Preview" 
                  style={{
                    maxWidth: '200px',
                    marginTop: '10px',
                    borderRadius: '5px',
                    border: '2px solid #4CAF50'
                  }}
                />
              </div>
            )}
          </div>
        )}
      </div>
      
      <button 
        type="button" 
        className="upload-btn" 
        onClick={openFileDialog}
      >
        Browse {type === 'mri' ? 'MRI Image' : 'Biomarker File'}
      </button>
    </div>
  );
};

export default FileUpload;