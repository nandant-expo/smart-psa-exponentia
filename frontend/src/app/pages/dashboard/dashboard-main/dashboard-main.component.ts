import { Component, OnInit } from '@angular/core';
import { ApiService } from 'src/app/common/services/api.service';

@Component({
  selector: 'app-dashboard-main',
  templateUrl: './dashboard-main.component.html',
  styleUrls: ['./dashboard-main.component.scss']
})
export class DashboardMainComponent implements OnInit {
  role = '';
  
  selectedCxoSheet = { name: 'Dashboard', code: 'dashboard' };
  selectedCategoryManagerSheet = { name: 'Dashboard', code: 'dashboard' };
  selectedContractManagerSheet = { name: 'Contracts', code: 'contracts' };
  
  cxoSheets = [
    { name: 'Dashboard', code: 'dashboard' },
    { name: 'Spending', code: 'spending' },
    { name: 'Payments', code: 'payments' },
    { name: 'Vendor Profile', code: 'vendorProfile' },
    { name: 'Contracts', code: 'contracts' }
  ];

  categoryManagerSheets = [
    { name: 'Dashboard', code: 'dashboard' },
    { name: 'Spending', code: 'spending' },
    { name: 'Payments', code: 'payments' },
    { name: 'Vendor Profile', code: 'vendorProfile' },
  ];

  contractManagerSheets = [
    { name: 'Contracts', code: 'contracts' }
  ];

  constructor(
    private readonly apiService: ApiService
  ) {}

  ngOnInit(): void {
    this.getRoleFromApi()
  }

  getRoleFromApi = () => {
    this.role = this.apiService.getRole() ?? '';
  }
}
