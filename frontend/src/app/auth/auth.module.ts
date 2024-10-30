import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { SharedModule } from '../shared/shared.module';
import { PasswordModule } from 'primeng/password';
import { AuthRoutingModule } from './auth-routing.module';
import { LoginComponent } from './login/login.component';
import { Footer1Component } from '../common/components/footer/footer1/footer1.component';

@NgModule({
  declarations: [
    LoginComponent,
    Footer1Component,
  ],
  imports: [
    CommonModule,
    SharedModule,
    PasswordModule,
    AuthRoutingModule
  ]
})
export class AuthModule { }
