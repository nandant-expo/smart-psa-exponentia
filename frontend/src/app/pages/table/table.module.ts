import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';

import { TableRoutingModule } from './table-routing.module';
import { TableMainComponent } from './table-main/table-main.component';
import { SharedModule } from 'src/app/shared/shared.module';


@NgModule({
  declarations: [
    TableMainComponent
  ],
  imports: [
    CommonModule,
    TableRoutingModule,
    SharedModule
  ]
})
export class TableModule { }
