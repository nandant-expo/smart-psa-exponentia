import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ChatAskAQuestionComponent } from './chat-ask-a-question.component';
import { ChatService } from '../services/chat.service';
import { HttpClientTestingModule } from '@angular/common/http/testing';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';

describe('ChatAskAQuestionComponent', () => {
  let component: ChatAskAQuestionComponent;
  let fixture: ComponentFixture<ChatAskAQuestionComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ ChatAskAQuestionComponent ],
      providers: [ ChatService ],
      imports: [ HttpClientTestingModule, FormsModule, ReactiveFormsModule ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(ChatAskAQuestionComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
