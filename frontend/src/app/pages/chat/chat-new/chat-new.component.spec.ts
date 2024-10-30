import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ChatNewComponent } from './chat-new.component';
import { HttpClientTestingModule } from '@angular/common/http/testing';
import { ToastModule } from 'primeng/toast';
import { ChatAskAQuestionComponent } from '../chat-ask-a-question/chat-ask-a-question.component';
import { FormsModule } from '@angular/forms';
import { ChatService } from '../services/chat.service';

describe('ChatNewComponent', () => {
  let component: ChatNewComponent;
  let fixture: ComponentFixture<ChatNewComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ ChatNewComponent, ChatAskAQuestionComponent ],
      imports: [ HttpClientTestingModule, ToastModule, FormsModule ],
      providers: [ ChatService ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(ChatNewComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
