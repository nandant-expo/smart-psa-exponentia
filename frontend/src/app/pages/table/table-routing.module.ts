import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { TableMainComponent } from './table-main/table-main.component';

const routes: Routes = [
  {
    path: '',
    component: TableMainComponent
  }
];

@NgModule({
  imports: [RouterModule.forChild(routes)],
  exports: [RouterModule]
})
export class TableRoutingModule { }
