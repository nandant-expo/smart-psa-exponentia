import { Component, OnDestroy, OnInit } from '@angular/core';
import { ComplianceService } from '../compliance.service';
import { Subscription } from 'rxjs';
import { MessageService } from 'primeng/api';

@Component({
  selector: 'app-compliance-main',
  templateUrl: './compliance-main.component.html',
  styleUrls: ['./compliance-main.component.scss']
})
export class ComplianceMainComponent implements OnInit, OnDestroy {
  tableRows = 9;
  invoiceIndex = 0;
  totalInvoices: string | number = '—';
  nonComplyingInvoices: string | number = '—';
  serachStr = '';
  loading = true;
  isResolved = false;
  isDialogVisible = false;
  updatedTill: string | Date = '—';
  invoiceData: any[] = []
  filteredInvoiceData: any[] = [];
  getInvoicesSubscription = new Subscription()
  deleteInvoiceSubscription = new Subscription()
  resolveInvoiceSubscription = new Subscription()

  constructor(
    private readonly complianceService: ComplianceService,
    private readonly messageService: MessageService
  ) {}

  ngOnInit(): void {
    this.getInvoicesFromApi();
  }

  getInvoicesFromApi = () => {
    this.getInvoicesSubscription =
      this.complianceService.getInvoices().subscribe({
        next: (res: any) => {
          this.loading = false;
          this.nonComplyingInvoices = res?.data?.non_complying_invoices ?? '—'
          this.totalInvoices = res?.data?.total_invoices ?? '—'
          this.updatedTill = res?.data?.update_till ?? new Date()
          this.invoiceData = res?.data?.invoices ?? []
          this.filteredInvoiceData =
            this.invoiceData?.filter((invoice: any) => invoice?.is_resolved === this.isResolved)

          this.messageService.add({
            severity: 'success',
            summary: 'Success',
            life: 5000,
            detail: res?.message ?? 'Invoices fetched successfully'
          });
        },
        error: (err: any) => {
          console.log('err ', err);
          this.messageService.add({
            severity: 'error',
            summary: 'Error',
            life: 5000,
            detail: 'Something went wrong while fetching invoices'
          });
        }
      })
  }


  toggleIsResolved = () => {
    this.isResolved = !this.isResolved;
    this.filteredInvoiceData =
          this.invoiceData?.filter((invoice: any) => invoice?.is_resolved === this.isResolved)
  }

  searchItem = (e: any) => {
    this.serachStr = e.target.value;
  }

  setTableRows = () => {
    if (window.innerHeight >= 800) {
      this.tableRows = 12;
    }
  }

  openDialog = (invoiceIndex: number) => {
    this.isDialogVisible = true;
    this.invoiceIndex = invoiceIndex;
  }

  resolveInvoiceFromApi = (invoiceData: any) => {
    this.resolveInvoiceSubscription =
      this.complianceService.resolveInvoice(invoiceData).subscribe({
        next: (res: any) => {
          this.messageService.add({
            severity: 'success',
            summary: 'Success',
            life: 5000,
            detail: res?.message ?? 'Invoice resolved successfully'
          });

          this.getInvoicesFromApi();
        },
        error: (err: any) => {
          console.log('err ', err);
          
          this.messageService.add({
            severity: 'error',
            summary: 'Error',
            life: 5000,
            detail: 'Something went wrong while resolving invoice'
          });
        }
      })
  }

  deleteInvoice = (invoiceData: any) => {
    this.deleteInvoiceSubscription =
      this.complianceService.deleteInvoice(invoiceData).subscribe({
        next: (res: any) => {
          this.messageService.add({
            severity: 'success',
            summary: 'Success',
            life: 5000,
            detail: res?.message ?? 'Invoice deleted successfully'
          });

          this.getInvoicesFromApi()
        },
        error: (err: any) => {
          console.log('err ', err);

          this.messageService.add({
            severity: 'error',
            summary: 'Error',
            life: 5000,
            detail: 'Something went wrong while deleting invoice'
          });
        }
      })
  }

  ngOnDestroy(): void {
    this.getInvoicesSubscription.unsubscribe();
    this.deleteInvoiceSubscription.unsubscribe();
    this.resolveInvoiceSubscription.unsubscribe();
  }
}
