import { Component, OnInit } from '@angular/core';

@Component({
  selector: 'app-table-main',
  templateUrl: './table-main.component.html',
  styleUrls: ['./table-main.component.scss']
})
export class TableMainComponent implements OnInit {
  tableRows = 9;
  serachStr = '';
  loading = false;
  currentDateTime = new Date();
  cleanTableData: any[] = [];
  uncleanTableData: any[] = [
    {
      "": "Instruction Descriptions",
      "file_name": "",
      "Contract Unique Identifier": "find the unique identifier for the contract, this is file name of the contract",
      "Contract Agreed Date": "Find date when the contract was agreed upon. Format as DD/MM/YYYY",
      "Automatic Alert Date": "FInd date set for automatic alerts related to the contract. Format as DD/MM/YYYY",
      "Country associated with contract": "Find full name of the country associated with the contract.",
      "Date of contract creation": "Find date when the contract record was created in the system. Format as DD/MM/YYYY",
      "Currency used in contract": "Find currency used in the contract.",
      "Effective contract date": "Find date when the contract becomes effective. Format as DD/MM/YYYY",
      "Date Reminder": "Find the reminder date for evergreen (automatically renewing) contracts (if applicable) else revert \"N.A\". Format as DD/MM/YYYY",
      "Type of contract": "The type of the contract from this list (Local) Contract, Group Master Agreement, Licence Agreement, Local Master Agreement, Participation Agreement, Service Agreement, and Statement of Work (SOW).",
      "Contract Failure Legal Jurisdiction": "Find legal jurisdiction under which the contract falls.",
      "Person/Entity Manager": "find person or entity operating or managing the contract.",
      "External contractor": "find the Name of the external contractor.",
      "total mmonetary value": "what is the total monetary value of the contract. Don’t include the symbol of the amount",
      "Contract Termination date": "what is the date when the contract was or will be terminated. Format as DD/MM/YYYY",
      "Date of termination/prolongation decision": "what is The date for termination or prolongation decision. Format as DD/MM/YYYY",
      "Timespan for termination process": "what is The timespan for the termination process. Mention in number of days",
      "type of termination": "what is The type of termination type (e.g., Autorenew, autoterminate, evergreen or other) if applicable.",
      "Permanent terms": "what are the Payment terms specified in  contract"
    },
    {
      "": "",
      "file_name": "321456723.pdf",
      "Contract Unique Identifier": 321456723,
      "Contract Agreed Date": "1/10/2024",
      "Automatic Alert Date": "1/11/2025",
      "Country associated with contract": "Ukraine",
      "Date of contract creation": "10/10/2024",
      "Currency used in contract": "UAH",
      "Effective contract date": "1/1/2025",
      "Date Reminder": "N.A",
      "Type of contract": "Service Agreement",
      "Contract Failure Legal Jurisdiction": "Kyiv, Ukraine",
      "Person/Entity Manager": "Urban Insurance Company",
      "External contractor": "ABC Reinsurance Ltd.",
      "total mmonetary value": 2500000,
      "Contract Termination date": "N/A",
      "Date of termination/prolongation decision": "1/1/2026",
      "Timespan for termination process": 365,
      "type of termination": "Automatic Renewal",
      "Permanent terms": "Quarterly payments"
    },
    {
      "": "",
      "file_name": "321456719.pdf",
      "Contract Unique Identifier": 321456719,
      "Contract Agreed Date": "10/10/2024",
      "Automatic Alert Date": "10/10/2025",
      "Country associated with contract": "Hungary",
      "Date of contract creation": "10/10/2024",
      "Currency used in contract": "HUF",
      "Effective contract date": "15/10/2024",
      "Date Reminder": "N.A",
      "Type of contract": "Service Agreement",
      "Contract Failure Legal Jurisdiction": "Budapest, Hungary",
      "Person/Entity Manager": "Anna Kovacs",
      "External contractor": "Istvan Toth",
      "total mmonetary value": 5000000,
      "Contract Termination date": "N.A",
      "Date of termination/prolongation decision": "15/10/2026",
      "Timespan for termination process": 730,
      "type of termination": "Automatic Renewal",
      "Permanent terms": "Quarterly, due within 30 days of invoice receipt"
    },
    {
      "": "",
      "file_name": "321456721.pdf",
      "Contract Unique Identifier": 321456721,
      "Contract Agreed Date": "10/10/2024",
      "Automatic Alert Date": "10/10/2025",
      "Country associated with contract": "Hungary",
      "Date of contract creation": "10/10/2024",
      "Currency used in contract": "HUF",
      "Effective contract date": "1/11/2024",
      "Date Reminder": "N.A",
      "Type of contract": "Service Agreement",
      "Contract Failure Legal Jurisdiction": "Budapest, Hungary",
      "Person/Entity Manager": "PointUrban Asset Management Ltd.",
      "External contractor": "Investment Solutions Group Ltd.",
      "total mmonetary value": 1500000,
      "Contract Termination date": "N/A",
      "Date of termination/prolongation decision": "N/A",
      "Timespan for termination process": 365,
      "type of termination": "Automatic Renewal",
      "Permanent terms": "Monthly payments, due on the last day of each month"
    },
    {
      "": "",
      "file_name": "321456715.pdf",
      "Contract Unique Identifier": 321456715,
      "Contract Agreed Date": "10/10/2024",
      "Automatic Alert Date": "N.A",
      "Country associated with contract": "Slovakia",
      "Date of contract creation": "N.A",
      "Currency used in contract": "Euro",
      "Effective contract date": "10/10/2024",
      "Date Reminder": "N.A",
      "Type of contract": "Risk Assessment Agreement",
      "Contract Failure Legal Jurisdiction": "Slovakia",
      "Person/Entity Manager": "Anna Novak",
      "External contractor": "SafeRisk Solutions, s.r.o.",
      "total mmonetary value": "N.A",
      "Contract Termination date": "N.A",
      "Date of termination/prolongation decision": "N.A",
      "Timespan for termination process": "N.A",
      "type of termination": "N.A",
      "Permanent terms": "Quarterly payments, 1.5% interest per month for late payments"
    },
    {
      "": "",
      "file_name": "321456713.pdf",
      "Contract Unique Identifier": 321456713,
      "Contract Agreed Date": "10/10/2024",
      "Automatic Alert Date": "10/10/2025",
      "Country associated with contract": "Czech Republic",
      "Date of contract creation": "10/10/2024",
      "Currency used in contract": "CZK",
      "Effective contract date": "15/10/2024",
      "Date Reminder": "N.A",
      "Type of contract": "Service Agreement",
      "Contract Failure Legal Jurisdiction": "Czech Republic",
      "Person/Entity Manager": "Urban Insurance CZ, a.s.",
      "External contractor": "Jan Novak",
      "total mmonetary value": 1000000,
      "Contract Termination date": "N.A",
      "Date of termination/prolongation decision": "N.A",
      "Timespan for termination process": 365,
      "type of termination": "Annual Renewal",
      "Permanent terms": "Quarterly Payments Due on the 1st of January, April, July, and October"
    },
    {
      "": "",
      "file_name": "321456717.pdf",
      "Contract Unique Identifier": "UT-SI-2024-001",
      "Contract Agreed Date": "5/10/2024",
      "Automatic Alert Date": "30/10/2024",
      "Country associated with contract": "Slovakia",
      "Date of contract creation": "10/10/2024",
      "Currency used in contract": "Euro",
      "Effective contract date": "15/10/2024",
      "Date Reminder": "N/A",
      "Type of contract": "Service Agreement",
      "Contract Failure Legal Jurisdiction": "Bratislava, Slovakia",
      "Person/Entity Manager": "Martin Novak",
      "External contractor": "Petra Horak",
      "total mmonetary value": 150000,
      "Contract Termination date": "N/A",
      "Date of termination/prolongation decision": "N/A",
      "Timespan for termination process": 30,
      "type of termination": "Mutual Agreement",
      "Permanent terms": "50% upfront payment upon contract signing, 25% upon completion of initial system integration, 25% upon final delivery and acceptance of all services"
    },
    {
      "": "",
      "file_name": "321456711.pdf",
      "Contract Unique Identifier": 321456711,
      "Contract Agreed Date": "10/10/2024",
      "Automatic Alert Date": "10/10/2025",
      "Country associated with contract": "Czech Republic",
      "Date of contract creation": "10/10/2024",
      "Currency used in contract": "EUR",
      "Effective contract date": "10/10/2024",
      "Date Reminder": "N.A",
      "Type of contract": "Licence Agreement",
      "Contract Failure Legal Jurisdiction": "Czech Republic",
      "Person/Entity Manager": "John Doe",
      "External contractor": "Tech Innovations LLC",
      "total mmonetary value": 50000,
      "Contract Termination date": "N/A",
      "Date of termination/prolongation decision": "10/10/2025",
      "Timespan for termination process": 30,
      "type of termination": "Breach of Contract",
      "Permanent terms": "50% upfront, 50% upon delivery of technology"
    },
    {
      "": "",
      "file_name": "32145679.pdf",
      "Contract Unique Identifier": 32145679,
      "Contract Agreed Date": "10/10/2024",
      "Automatic Alert Date": "10/11/2024",
      "Country associated with contract": "Germany",
      "Date of contract creation": "1/10/2024",
      "Currency used in contract": "Euro",
      "Effective contract date": "15/10/2024",
      "Date Reminder": "N.A",
      "Type of contract": "Property Insurance Agreement",
      "Contract Failure Legal Jurisdiction": "Frankfurt, Germany",
      "Person/Entity Manager": "Urban Sachversicherung AG",
      "External contractor": "Max Mustermann",
      "total mmonetary value": 500000,
      "Contract Termination date": "N.A",
      "Date of termination/prolongation decision": "15/10/2025",
      "Timespan for termination process": 30,
      "type of termination": "With Cause",
      "Permanent terms": "Payments are due within 30 days of invoice receipt. Payments will be made via bank transfer in Euro (EUR)."
    },
    {
      "": "",
      "file_name": "32145677.pdf",
      "Contract Unique Identifier": 32145677,
      "Contract Agreed Date": "10/10/2024",
      "Automatic Alert Date": "N/A",
      "Country associated with contract": "Germany",
      "Date of contract creation": "10/10/2024",
      "Currency used in contract": "EUR",
      "Effective contract date": "1/1/2024",
      "Date Reminder": "N/A",
      "Type of contract": "Service Agreement",
      "Contract Failure Legal Jurisdiction": "Berlin, Germany",
      "Person/Entity Manager": "Urban Personenversicherung AG",
      "External contractor": "John Doe",
      "total mmonetary value": 100000,
      "Contract Termination date": "N/A",
      "Date of termination/prolongation decision": "N/A",
      "Timespan for termination process": "N/A",
      "type of termination": "N/A",
      "Permanent terms": "Monthly"
    },
    {
      "": "",
      "file_name": "321456722.pdf",
      "Contract Unique Identifier": 321456722,
      "Contract Agreed Date": "10/10/2024",
      "Automatic Alert Date": "N/A",
      "Country associated with contract": "Germany",
      "Date of contract creation": "10/10/2024",
      "Currency used in contract": "EUR",
      "Effective contract date": "1/11/2024",
      "Date Reminder": "N/A",
      "Type of contract": "Life Insurance Policy",
      "Contract Failure Legal Jurisdiction": "Germany",
      "Person/Entity Manager": "Urban Personenversicherung AG",
      "External contractor": "[Insured Name]",
      "total mmonetary value": 200000,
      "Contract Termination date": "N/A",
      "Date of termination/prolongation decision": "N/A",
      "Timespan for termination process": "N/A",
      "type of termination": "N/A",
      "Permanent terms": "Annual"
    },
    {
      "": "",
      "file_name": "32145672.pdf",
      "Contract Unique Identifier": 32145672,
      "Contract Agreed Date": "15/09/2024",
      "Automatic Alert Date": "1/9/2024",
      "Country associated with contract": "Austria",
      "Date of contract creation": "1/10/2024",
      "Currency used in contract": "EUR",
      "Effective contract date": "1/10/2024",
      "Date Reminder": "N.A",
      "Type of contract": "Service Agreement",
      "Contract Failure Legal Jurisdiction": "Austria",
      "Person/Entity Manager": "Johann Mller",
      "External contractor": "TechSol Consulting Ltd.",
      "total mmonetary value": 250000,
      "Contract Termination date": "31/12/2024",
      "Date of termination/prolongation decision": "31/01/2025",
      "Timespan for termination process": 30,
      "type of termination": "Termination for Convenience",
      "Permanent terms": "30% on Design, 40% on Development, 30% on Final Delivery"
    },
    {
      "": "",
      "file_name": "321456720.pdf",
      "Contract Unique Identifier": 321456720,
      "Contract Agreed Date": "10/10/2024",
      "Automatic Alert Date": "10/10/2025",
      "Country associated with contract": "Hungary",
      "Date of contract creation": "1/10/2024",
      "Currency used in contract": "HUF",
      "Effective contract date": "1/11/2024",
      "Date Reminder": "N.A",
      "Type of contract": "Service Agreement",
      "Contract Failure Legal Jurisdiction": "Budapest, Hungary",
      "Person/Entity Manager": "Urban Pension Fund Kft.",
      "External contractor": "Global Wealth Management Ltd.",
      "total mmonetary value": 1500000,
      "Contract Termination date": "N/A",
      "Date of termination/prolongation decision": "N/A",
      "Timespan for termination process": "N/A",
      "type of termination": "N/A",
      "Permanent terms": "Monthly contributions of 10% of gross salary"
    },
    {
      "": "",
      "file_name": "321456718.pdf",
      "Contract Unique Identifier": 321456718,
      "Contract Agreed Date": "15/01/2024",
      "Automatic Alert Date": "15/01/2025",
      "Country associated with contract": "Hungary",
      "Date of contract creation": "10/10/2024",
      "Currency used in contract": "HUF",
      "Effective contract date": "1/11/2024",
      "Date Reminder": "N.A",
      "Type of contract": "Local Contract",
      "Contract Failure Legal Jurisdiction": "Budapest, Hungary",
      "Person/Entity Manager": "Mr. László Kovács",
      "External contractor": "Ms. Anna Szabo",
      "total mmonetary value": 1500000,
      "Contract Termination date": "N/A",
      "Date of termination/prolongation decision": "N/A",
      "Timespan for termination process": 365,
      "type of termination": "Annual Renewal",
      "Permanent terms": "Monthly Installments"
    },
    {
      "": "",
      "file_name": "321456716.pdf",
      "Contract Unique Identifier": 321456716,
      "Contract Agreed Date": "5/10/2024",
      "Automatic Alert Date": "15/11/2024",
      "Country associated with contract": "Slovakia",
      "Date of contract creation": "10/10/2024",
      "Currency used in contract": "EUR",
      "Effective contract date": "15/10/2024",
      "Date Reminder": "N/A",
      "Type of contract": "Service Agreement",
      "Contract Failure Legal Jurisdiction": "Bratislava, Slovakia",
      "Person/Entity Manager": "John Doe",
      "External contractor": "Global IT Solutions, Inc.",
      "total mmonetary value": 150000,
      "Contract Termination date": "N/A",
      "Date of termination/prolongation decision": "15/04/2025",
      "Timespan for termination process": 180,
      "type of termination": "Mutual Agreement",
      "Permanent terms": "Net 30 days"
    },
    {
      "": "",
      "file_name": "321456714.pdf",
      "Contract Unique Identifier": 321456714,
      "Contract Agreed Date": "20/12/2023",
      "Automatic Alert Date": "30/11/2024",
      "Country associated with contract": "Slovakia",
      "Date of contract creation": "10/12/2023",
      "Currency used in contract": "Euros",
      "Effective contract date": "15/01/2024",
      "Date Reminder": "N.A",
      "Type of contract": "Reinsurance Agreement",
      "Contract Failure Legal Jurisdiction": "District Court of Bratislava I, Slovakia",
      "Person/Entity Manager": "Ms. Katarina Novakova",
      "External contractor": "Global Reinsurance Partners Ltd.",
      "total mmonetary value": 15000000,
      "Contract Termination date": "14/01/2025",
      "Date of termination/prolongation decision": "31/12/2024",
      "Timespan for termination process": 90,
      "type of termination": "mutual agreement, breach of terms, insolvency, non-payment of premiums",
      "Permanent terms": "quarterly installments"
    },
    {
      "": "",
      "file_name": "321456712.pdf",
      "Contract Unique Identifier": "UT-2024-001",
      "Contract Agreed Date": "10/10/2024",
      "Automatic Alert Date": "10/10/2025",
      "Country associated with contract": "Czech Republic",
      "Date of contract creation": "10/10/2024",
      "Currency used in contract": "CZK",
      "Effective contract date": "1/11/2024",
      "Date Reminder": "N.A",
      "Type of contract": "Service Agreement",
      "Contract Failure Legal Jurisdiction": "Czech Republic",
      "Person/Entity Manager": "Urban Technologies CZ, s.r.o.",
      "External contractor": "TechSupport Services s.r.o.",
      "total mmonetary value": 1200000,
      "Contract Termination date": "N/A",
      "Date of termination/prolongation decision": "N/A",
      "Timespan for termination process": 365,
      "type of termination": "Upon Notice",
      "Permanent terms": "Monthly in arrears"
    },
    {
      "": "",
      "file_name": "321456710.pdf",
      "Contract Unique Identifier": "CC-2024-001",
      "Contract Agreed Date": "10/10/2024",
      "Automatic Alert Date": "N.A",
      "Country associated with contract": "Germany",
      "Date of contract creation": "1/10/2024",
      "Currency used in contract": "Euro",
      "Effective contract date": "1/11/2024",
      "Date Reminder": "N.A",
      "Type of contract": "Service Agreement",
      "Contract Failure Legal Jurisdiction": "Germany",
      "Person/Entity Manager": "Urban Callcenter GmbH",
      "External contractor": "Tech Solutions Inc.",
      "total mmonetary value": 250000,
      "Contract Termination date": "31/10/2025",
      "Date of termination/prolongation decision": "30/11/2025",
      "Timespan for termination process": 30,
      "type of termination": "Mutual Agreement",
      "Permanent terms": "Payment due within 30 days of invoice receipt"
    },
    {
      "": "",
      "file_name": "32145676.pdf",
      "Contract Unique Identifier": 32145676,
      "Contract Agreed Date": "15/09/2024",
      "Automatic Alert Date": "1/9/2024",
      "Country associated with contract": "United States",
      "Date of contract creation": "1/10/2024",
      "Currency used in contract": "USD",
      "Effective contract date": "1/10/2024",
      "Date Reminder": "N.A",
      "Type of contract": "Service Agreement",
      "Contract Failure Legal Jurisdiction": "California",
      "Person/Entity Manager": "Digitech Solutions Inc.",
      "External contractor": "Innovatech Group GmbH",
      "total mmonetary value": 300000,
      "Contract Termination date": "31/12/2024",
      "Date of termination/prolongation decision": "31/01/2025",
      "Timespan for termination process": 30,
      "type of termination": "Termination for Convenience",
      "Permanent terms": "30% on Initial Design, 40% on Development, 30% on Final Delivery"
    },
    {
      "": "",
      "file_name": "32145673.pdf",
      "Contract Unique Identifier": 32145673,
      "Contract Agreed Date": "15/09/2024",
      "Automatic Alert Date": "1/9/2024",
      "Country associated with contract": "United States",
      "Date of contract creation": "1/10/2024",
      "Currency used in contract": "USD",
      "Effective contract date": "1/10/2024",
      "Date Reminder": "N.A",
      "Type of contract": "Service Agreement",
      "Contract Failure Legal Jurisdiction": "California",
      "Person/Entity Manager": "Digitech Solutions Inc.",
      "External contractor": "Innovatech Group GmbH",
      "total mmonetary value": 300000,
      "Contract Termination date": "31/12/2024",
      "Date of termination/prolongation decision": "31/01/2025",
      "Timespan for termination process": 30,
      "type of termination": "Termination for Convenience",
      "Permanent terms": "30% on Initial Design, 40% on Development, 30% on Final Delivery"
    },
    {
      "": "",
      "file_name": "32145671.pdf",
      "Contract Unique Identifier": 32145671,
      "Contract Agreed Date": "9/10/2024",
      "Automatic Alert Date": "9/11/2024",
      "Country associated with contract": "Germany",
      "Date of contract creation": "9/10/2024",
      "Currency used in contract": "Euro",
      "Effective contract date": "15/10/2024",
      "Date Reminder": "N.A",
      "Type of contract": "Service Agreement",
      "Contract Failure Legal Jurisdiction": "German Law",
      "Person/Entity Manager": "Urban Real Estate Management GmbH",
      "External contractor": "Tech Solutions Inc.",
      "total mmonetary value": 250000,
      "Contract Termination date": "15/10/2025",
      "Date of termination/prolongation decision": "15/10/2026",
      "Timespan for termination process": 30,
      "type of termination": "Autorenew",
      "Permanent terms": "30% upon signing, 40% upon delivery, 30% upon completion"
    }
  ]

  ngOnInit(): void {
    this.cleanTableData = this.uncleanTableData.slice(1)
    this.setTableRows();
  }

  searchItem = (e: any) => {
    this.serachStr = e.target.value;
  }

  setTableRows = () => {
    if (window.innerHeight >= 800) {
      this.tableRows = 12;
    }
  }
}
