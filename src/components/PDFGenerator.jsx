import React from 'react';
import { jsPDF } from 'jspdf';
import { useAuth, useNotification } from '../App';

const PDFGenerator = ({ results, carePlan, mriImage, selectedPatient }) => {
  const { user } = useAuth();
  const { showNotification } = useNotification();

  const generatePDF = async () => {
    if (!results) {
      showNotification('No analysis results available for PDF generation', 'error');
      return;
    }

    if (!selectedPatient) {
      showNotification('No patient selected for PDF generation', 'error');
      return;
    }

    try {
      const doc = new jsPDF();
      const pageWidth = 210;
      const pageHeight = 297;
      const margin = 15;
      const headerHeight = 40;

      // Get patient data from selected patient
      const patientData = getPatientDataFromSelected(selectedPatient);

      // Add first page with header (including logo)
      await addPageHeaderAsync(doc, pageWidth, margin);

      // Patient details table
      addPatientDetails(doc, pageWidth, margin, headerHeight, patientData);

      // Analysis Results section
      await addAnalysisResultsAsync(doc, pageWidth, margin, headerHeight);

      // Add second page for Plan of Care
      doc.addPage();
      await addPageHeaderAsync(doc, pageWidth, margin);
      addPlanOfCare(doc, pageWidth, pageHeight, margin, headerHeight, patientData);

      // Save PDF with patient name
      const timestamp = new Date().toISOString().slice(0, 10);
      const fileName = `Medical_AI_Report_${patientData.name.replace(/\s+/g, '_')}_${timestamp}.pdf`;
      doc.save(fileName);

      showNotification(`PDF report generated for ${patientData.name}!`, 'success');

    } catch (error) {
      console.error('PDF generation error:', error);
      showNotification('PDF generation failed', 'error');
    }
  };

  const addPageHeaderAsync = async (doc, pageWidth, margin) => {
    // Add main border
    doc.rect(10, 10, pageWidth - 20, 277);

    // Add logo
    await addLogoToPDF(doc, margin, 20);

    // Add title box
    const titleBoxX = 70;
    const titleBoxWidth = 125;
    doc.rect(titleBoxX, 15, titleBoxWidth, 20);

    doc.setFontSize(11);
    doc.setFont(undefined, 'bold');
    doc.setTextColor(0, 0, 0);
    const titleText = 'Medical AI - Dementia Analysis System';
    const titleWidth = doc.getTextWidth(titleText);
    const titleX = titleBoxX + (titleBoxWidth - titleWidth) / 2;
    doc.text(titleText, titleX, 27);

    // Main title "REPORT"
    doc.setFontSize(18);
    doc.setFont(undefined, 'bold');
    const reportTitle = 'REPORT';
    const reportTitleWidth = doc.getTextWidth(reportTitle);
    doc.text(reportTitle, (pageWidth - reportTitleWidth) / 2, 50);

    // Line under REPORT
    doc.line(margin, 55, pageWidth - margin, 55);
  };

  const addLogoToPDF = (doc, x, y) => {
    return new Promise((resolve) => {
      // Create an image element to load the logo
      const img = new Image();
      img.crossOrigin = 'anonymous';
      
      img.onload = function() {
        try {
          // Create a canvas to convert the image to base64
          const canvas = document.createElement('canvas');
          const ctx = canvas.getContext('2d');
          
          // Set canvas size
          canvas.width = img.width;
          canvas.height = img.height;
          
          // Draw image to canvas
          ctx.drawImage(img, 0, 0);
          
          // Get base64 data
          const dataURL = canvas.toDataURL('image/png');
          
          // Add to PDF (logo area: 50x20)
          const logoWidth = 45;
          const logoHeight = 15;
          doc.addImage(dataURL, 'PNG', x, y, logoWidth, logoHeight);
          
          resolve();
        } catch (error) {
          console.error('Error processing logo:', error);
          // Fallback to text
          addLogoFallback(doc, x, y);
          resolve();
        }
      };
      
      img.onerror = function() {
        console.error('Error loading logo from /logo.png');
        // Fallback to text
        addLogoFallback(doc, x, y);
        resolve();
      };
      
      // Load the logo from public folder
      img.src = '/logo.png';
    });
  };

  const addLogoFallback = (doc, x, y) => {
    // Fallback text logo if image fails to load
    doc.setFontSize(16);
    doc.setFont(undefined, 'bold');
    doc.setTextColor(102, 126, 234);
    doc.text('TietoEvry', x, y + 12);
    doc.setTextColor(0, 0, 0);
  };

  const addPatientDetails = (doc, pageWidth, margin, headerHeight, patientData) => {
    const detailsY = headerHeight + 25;
    const colWidth = (pageWidth - 2 * margin) / 4;

    // First row
    doc.rect(margin, detailsY, colWidth, 10);
    doc.rect(margin + colWidth, detailsY, colWidth, 10);
    doc.rect(margin + 2 * colWidth, detailsY, colWidth, 10);
    doc.rect(margin + 3 * colWidth, detailsY, colWidth, 10);

    doc.setFontSize(9);
    doc.setFont(undefined, 'normal');
    doc.text('Report Date:', margin + 2, detailsY + 7);

    const now = new Date();
    const formattedDate = `${String(now.getDate()).padStart(2, '0')}/${String(now.getMonth() + 1).padStart(2, '0')}/${now.getFullYear()}`;
    const formattedTime = `${String(now.getHours()).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')}`;

    doc.text(`${formattedDate} ${formattedTime}`, margin + colWidth + 2, detailsY + 7);
    doc.text('Physician:', margin + 2 * colWidth + 2, detailsY + 7);
    doc.text(`Dr. ${user?.firstName || user?.username || 'Unknown'}`, margin + 3 * colWidth + 2, detailsY + 7);

    // Second row
    const row2Y = detailsY + 10;
    doc.rect(margin, row2Y, colWidth, 10);
    doc.rect(margin + colWidth, row2Y, colWidth, 10);
    doc.rect(margin + 2 * colWidth, row2Y, colWidth, 10);
    doc.rect(margin + 3 * colWidth, row2Y, colWidth, 10);

    doc.text('Patient Name:', margin + 2, row2Y + 7);
    doc.text(patientData.name, margin + colWidth + 2, row2Y + 7);
    doc.text('Patient ID:', margin + 2 * colWidth + 2, row2Y + 7);
    doc.text(patientData.id, margin + 3 * colWidth + 2, row2Y + 7);

    // Third row with additional patient details
    const row3Y = row2Y + 10;
    doc.rect(margin, row3Y, colWidth, 10);
    doc.rect(margin + colWidth, row3Y, colWidth, 10);
    doc.rect(margin + 2 * colWidth, row3Y, colWidth, 10);
    doc.rect(margin + 3 * colWidth, row3Y, colWidth, 10);

    doc.text('Age:', margin + 2, row3Y + 7);
    doc.text(patientData.age, margin + colWidth + 2, row3Y + 7);
    doc.text('Gender:', margin + 2 * colWidth + 2, row3Y + 7);
    doc.text(patientData.gender, margin + 3 * colWidth + 2, row3Y + 7);

    // Fourth row with medical condition
    const row4Y = row3Y + 10;
    doc.rect(margin, row4Y, colWidth * 2, 10);
    doc.rect(margin + 2 * colWidth, row4Y, colWidth * 2, 10);

    doc.text('Medical Condition:', margin + 2, row4Y + 7);
    doc.text(patientData.condition, margin + 2 * colWidth + 2, row4Y + 7);
  };

  const addAnalysisResultsAsync = async (doc, pageWidth, margin, headerHeight) => {
    // Analysis Results header (adjusted for additional patient rows)
    doc.setFontSize(12);
    doc.setFont(undefined, 'bold');
    doc.text('Analysis Results :', margin, headerHeight + 70);

    // MRI and Biomarkers table (adjusted position)
    const tableY = headerHeight + 80;
    const mriWidth = 70;
    const bioWidth = pageWidth - 2 * margin - mriWidth;
    const tableHeight = 70;

    // MRI section
    doc.rect(margin, tableY, mriWidth, 15);
    doc.setFontSize(12);
    doc.setFont(undefined, 'bold');
    doc.text('MRI', margin + mriWidth / 2 - 8, tableY + 10);

    doc.rect(margin, tableY + 15, mriWidth, tableHeight - 15);

    // Add MRI image
    await addMRIImageToPDFAsync(doc, margin, tableY, mriWidth, tableHeight);

    // Biomarkers section
    doc.rect(margin + mriWidth, tableY, bioWidth, 15);
    doc.setFontSize(12);
    doc.setFont(undefined, 'bold');
    doc.text('BIOMARKERS', margin + mriWidth + bioWidth / 2 - 20, tableY + 10);

    doc.rect(margin + mriWidth, tableY + 15, bioWidth, tableHeight - 15);

    // Add biomarkers table
    addBiomarkersTable(doc, margin + mriWidth, tableY, bioWidth, tableHeight);

    // Predicted Stage
    const stageY = tableY + tableHeight + 20;
    doc.rect(margin + 30, stageY, 100, 15);
    doc.setFontSize(10);
    doc.setFont(undefined, 'normal');
    doc.text('Predicted Stage:', margin + 35, stageY + 10);
    doc.setFont(undefined, 'bold');
    doc.text(results.predicted_stage, margin + 85, stageY + 10);

    // Estimated Progression
    const progY = stageY + 25;
    doc.rect(margin + 30, progY, 100, 15);
    doc.setFontSize(10);
    doc.setFont(undefined, 'normal');
    doc.text('Estimated Progression:', margin + 35, progY + 10);
    doc.setFont(undefined, 'bold');
    doc.text(`${results.progression_months.toFixed(1)} months`, margin + 95, progY + 10);

    
  };

  const addMRIImageToPDFAsync = (doc, x, y, width, height) => {
    return new Promise((resolve) => {
      if (!mriImage) {
        doc.setFontSize(10);
        doc.setFont(undefined, 'normal');
        doc.text('--IMAGE--', x + width / 2 - 12, y + height / 2 + 5);
        resolve();
        return;
      }

      const reader = new FileReader();
      reader.onload = function (e) {
        try {
          const imageFormat = mriImage.type.includes('png') ? 'PNG' : 'JPEG';
          const imageX = x + 5;
          const imageY = y + 20;
          const imageWidth = width - 10;
          const imageHeight = height - 25;

          doc.addImage(e.target.result, imageFormat, imageX, imageY, imageWidth, imageHeight);
        } catch (error) {
          console.error('Error adding MRI image to PDF:', error);
          doc.setFontSize(8);
          doc.text('Image Error', x + 10, y + height / 2);
        }
        resolve();
      };

      reader.onerror = function () {
        console.error('Error reading MRI file');
        doc.setFontSize(8);
        doc.text('Read Error', x + 10, y + height / 2);
        resolve();
      };

      reader.readAsDataURL(mriImage);
    });
  };

  const addBiomarkersTable = (doc, x, y, width, height) => {
    if (!results || !results.biomarkers) {
      doc.setFontSize(10);
      doc.text('No biomarker data', x + 10, y + 30);
      return;
    }

    const bioStartX = x + 5;
    const bioStartY = y + 25;
    const colWidths = [45, 25, 20];

    // Table headers
    doc.setFontSize(8);
    doc.setFont(undefined, 'bold');

    doc.rect(bioStartX, bioStartY - 5, colWidths[0], 8);
    doc.rect(bioStartX + colWidths[0], bioStartY - 5, colWidths[1], 8);
    doc.rect(bioStartX + colWidths[0] + colWidths[1], bioStartY - 5, colWidths[2], 8);

    doc.text('Biomarker', bioStartX + 2, bioStartY);
    doc.text('Value', bioStartX + colWidths[0] + 2, bioStartY);
    doc.text('Unit', bioStartX + colWidths[0] + colWidths[1] + 2, bioStartY);

    // Table data
    doc.setFontSize(7);
    doc.setFont(undefined, 'normal');
    let yPos = bioStartY + 8;

    Object.entries(results.biomarkers).forEach(([biomarker, value]) => {
      const shortName = biomarker.replace('Plasma ', '').replace('Amyloid Beta', 'AB');

      doc.rect(bioStartX, yPos - 5, colWidths[0], 6);
      doc.rect(bioStartX + colWidths[0], yPos - 5, colWidths[1], 6);
      doc.rect(bioStartX + colWidths[0] + colWidths[1], yPos - 5, colWidths[2], 6);

      doc.text(shortName, bioStartX + 2, yPos);
      doc.text(value.toFixed(3), bioStartX + colWidths[0] + 2, yPos);
      doc.text(getBiomarkerUnit(biomarker), bioStartX + colWidths[0] + colWidths[1] + 2, yPos);

      yPos += 6;
    });
  };

  const addPlanOfCare = (doc, pageWidth, pageHeight, margin, headerHeight, patientData) => {
    const careY = headerHeight + 25;
    doc.setFontSize(12);
    doc.setFont(undefined, 'bold');
    doc.text(`Plan of Care for ${patientData.name}:`, margin, careY);

    doc.setFontSize(9);
    doc.setFont(undefined, 'normal');
    let pointY = careY + 15;

    if (carePlan && carePlan.length > 0) {
      carePlan.forEach((point, index) => {
        // Add numbered bullet point
        doc.text(`${index + 1}.`, margin + 5, pointY);
        const maxWidth = pageWidth - 2 * margin - 15;
        const lines = doc.splitTextToSize(point, maxWidth);
        doc.text(lines, margin + 15, pointY);
        pointY += Math.max(lines.length * 4, 8) + 5;
      });
    } else {
      doc.text('• No care plan available', margin + 5, pointY);
    }

    // Signature section
    addSignature(doc, pageWidth, pageHeight, margin);
  };

  const addSignature = (doc, pageWidth, pageHeight, margin) => {
    const sigY = pageHeight - 50;
    doc.setFontSize(9);
    doc.setFont(undefined, 'normal');
    doc.text('Signature of the Physician', pageWidth - 70, sigY);

    doc.line(pageWidth - 70, sigY + 10, pageWidth - margin, sigY + 10);

    doc.setFontSize(10);
    doc.setFont(undefined, 'bold');
    doc.text(`Dr. ${user?.firstName || user?.username || 'Unknown'}`, pageWidth - 70, sigY + 20);
  };

  const getBiomarkerUnit = (biomarker) => {
    const units = {
      'P-Tau 181': 'pg/mL',
      'P-Tau 217': 'pg/mL',
      'P-Tau 231': 'pg/mL',
      'Amyloid Beta 42': 'pg/mL',
      'Amyloid Beta 40': 'pg/mL',
      'AB42/AB40': 'ratio'
    };
    return units[biomarker] || '';
  };

  const getPatientDataFromSelected = (patient) => {
    const firstName = patient?.first_name || 'Unknown';
    const lastName = patient?.last_name || 'Patient';
    const patientId = patient?.patient_id_number || patient?.id || 'N/A';
    
    // Calculate age from date of birth
    let age = 'Unknown';
    if (patient?.date_of_birth) {
      const birthDate = new Date(patient.date_of_birth);
      const today = new Date();
      age = today.getFullYear() - birthDate.getFullYear();
      const monthDiff = today.getMonth() - birthDate.getMonth();
      if (monthDiff < 0 || (monthDiff === 0 && today.getDate() < birthDate.getDate())) {
        age--;
      }
      age = age.toString();
    }

    return {
      name: `${firstName} ${lastName}`,
      id: patientId,
      age: age,
      gender: patient?.gender || 'Not specified',
      condition: patient?.medical_condition || 'Dementia Analysis'
    };
  };

  return (
    <button 
      className="generate-pdf-btn" 
      onClick={generatePDF}
      style={{ marginLeft: '15px' }}
      disabled={!selectedPatient}
      title={!selectedPatient ? 'Please select a patient first' : 'Generate PDF Report'}
    >
       Generate PDF Report
    </button>
  );
};

export default PDFGenerator;