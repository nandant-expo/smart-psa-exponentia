import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';

import { ComplianceRoutingModule } from './compliance-routing.module';
import { ComplianceMainComponent } from './compliance-main/compliance-main.component';
import { SharedModule } from 'src/app/shared/shared.module';
import { MessageService } from 'primeng/api';


@NgModule({
  declarations: [
    ComplianceMainComponent
  ],
  imports: [
    CommonModule,
    ComplianceRoutingModule,
    SharedModule
  ],
  providers: [
    MessageService
  ]
})
export class ComplianceModule { }
