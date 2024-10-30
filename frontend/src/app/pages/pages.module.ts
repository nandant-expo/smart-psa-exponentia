import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { LayoutComponent } from './layout/layout.component';
import { SharedModule } from '../shared/shared.module';
import { RouterModule, Routes } from '@angular/router';
import { AuthGuard } from '../common/guards/auth.guard';
import { ChatService } from './chat/services/chat.service';

const routes: Routes = [
  { 
    path: '', 
    canActivate: [AuthGuard],
    component: LayoutComponent,
    children:[
      {
        path: '',
        redirectTo: '/dashboard',
        pathMatch: 'full'
      },
      {
        path: 'chat',
        loadChildren: () => import('./chat/chat.module').then(m => m.ChatModule)
      },
      {
        path: 'compliance',
        loadChildren: () => import('./compliance/compliance.module').then(m => m.ComplianceModule)
      },
      {
        path: 'dashboard',
        loadChildren: () => import('./dashboard/dashboard.module').then(m => m.DashboardModule)
      },
      {
        path: 'structure-data-analysis',
        loadChildren: () => import('./chat-databricks-agent/chat-databricks-agent.module').then(m => m.ChatDatabricksAgentModule )
      },
      {
        path: 'table',
        loadChildren: () => import('./table/table.module').then(m => m.TableModule )
      }
    ]
  }
]
@NgModule({
  declarations: [
    LayoutComponent
  ],
  imports: [
    CommonModule,
    SharedModule,
    RouterModule.forChild(routes)
  ],
  providers: [
    ChatService
  ]
})
export class PagesModule { }
