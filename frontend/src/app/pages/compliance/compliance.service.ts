import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { environment } from 'src/environments/environment';

@Injectable({
  providedIn: 'root'
})
export class ComplianceService {

  constructor(
    private readonly http: HttpClient
  ) {}

  getInvoices = () => {
    return this.http.get(`${environment.apiendpoint}fetch_invoice`)
  }

  resolveInvoice = (formData: any) => {
    const invoiceId = formData?.invoice_id ?? '';

    return this.http.post(
      `${environment.apiendpoint}invoices_status/${invoiceId}`,
      { 'invoice_id': invoiceId }
    )
  }

  deleteInvoice = (formData: any) => {
    const invoiceId = formData?.invoice_id ?? '';
    
    return this.http.delete(`${environment.apiendpoint}delete_invoice/${invoiceId}`)
  }
}
