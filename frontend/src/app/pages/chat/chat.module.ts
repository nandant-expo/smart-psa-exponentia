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
import { ResizerThreeComponent } from './resizer-three/resizer-three.component';
import { ExistingFileComponent } from './existing-file/existing-file.component';
import { VideoWidgetComponent } from './video-widget/video-widget.component';
import { MessageService } from 'primeng/api';


const routes: Routes = [
  { 
    path: '', 
    component: MainComponent,
    children: [
      { path: '', component: ChatNewComponent},
      { path: 'resizer', component: ResizerThreeComponent},
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
    ResizerThreeComponent,
    ExistingFileComponent,
    VideoWidgetComponent,
  ],
  imports: [
    CommonModule,
    SharedModule,
    RouterModule.forChild(routes),
    InputTextModule,
    CheckboxModule,
    RadioButtonModule
  ],
  providers: [
    ChatService,
    MessageService
  ]
})
export class ChatModule { }
