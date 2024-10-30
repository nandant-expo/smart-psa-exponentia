import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { SharedModule } from '../../shared/shared.module';
import { RouterModule, Routes } from '@angular/router';
import { ChatComponent } from './chat/chat.component';
import { ChatHistoryListComponent } from './chat-history-list/chat-history-list.component';
import { MainComponent } from './main/main.component';
import { ChatMessageListComponent } from './chat-message-list/chat-message-list.component';
import { ChatNewComponent } from './chat-new/chat-new.component';
import { ChatAskAQuestionComponent } from './chat-ask-a-question/chat-ask-a-question.component';
import { InputTextModule } from 'primeng/inputtext';
import { CheckboxModule } from 'primeng/checkbox';
import { RadioButtonModule } from 'primeng/radiobutton';
import { ChatService } from './services/chat.service';
import { ExistingFileComponent } from './existing-file/existing-file.component';
import * as PlotlyJS from 'plotly.js-dist-min';
import { PlotlyModule } from 'angular-plotly.js';
import { HighlightService } from './services/highlight.service';
PlotlyModule.plotlyjs = PlotlyJS;

const routes: Routes = [
  { 
    path: '', 
    component: MainComponent,
    children: [
      { path: '', component: ChatNewComponent},
      { path: 'c/:id/:type', component: ChatComponent},
    ]
  
  }

]

@NgModule({
  declarations: [
    ChatComponent,
    ChatHistoryListComponent,
    MainComponent,
    ChatMessageListComponent,
    ChatNewComponent,
    ChatAskAQuestionComponent,
    ExistingFileComponent,
  ],
  imports: [
    CommonModule,
    SharedModule,
    RouterModule.forChild(routes),
    InputTextModule,
    CheckboxModule,
    RadioButtonModule,
    PlotlyModule
  ],
  providers:[ChatService, HighlightService]
})
export class ChatDatabricksAgentModule { }
