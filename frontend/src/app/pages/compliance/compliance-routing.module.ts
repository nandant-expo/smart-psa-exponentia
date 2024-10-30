import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { ComplianceMainComponent } from './compliance-main/compliance-main.component';

const routes: Routes = [
  {
    path: '',
    component: ComplianceMainComponent
  }
];

@NgModule({
  imports: [RouterModule.forChild(routes)],
  exports: [RouterModule]
})
export class ComplianceRoutingModule { }
