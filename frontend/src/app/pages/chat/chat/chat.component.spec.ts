import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ChatComponent } from './chat.component';
import { ChatAskAQuestionComponent } from '../chat-ask-a-question/chat-ask-a-question.component'
import { ChatService } from '../services/chat.service';

import { RouterTestingModule } from '@angular/router/testing';

import { SkeletonModule } from 'primeng/skeleton';
import { FormsModule } from '@angular/forms';
import { DialogModule } from 'primeng/dialog';
import { HttpClientTestingModule } from '@angular/common/http/testing';

describe('ChatComponent', () => {
  let component: ChatComponent;
  let fixture: ComponentFixture<ChatComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [
        HttpClientTestingModule,
        RouterTestingModule,
        SkeletonModule,
        FormsModule,
        DialogModule
      ],
      declarations: [ ChatComponent ,ChatAskAQuestionComponent],
      providers:[ChatService],
    })
    .compileComponents();

    fixture = TestBed.createComponent(ChatComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
