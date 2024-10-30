import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ChatAskAQuestionComponent } from './chat-ask-a-question.component';
import { HttpClientTestingModule } from '@angular/common/http/testing';
import { FormsModule } from '@angular/forms';
import { ChatService } from '../services/chat.service';

describe('ChatAskAQuestionComponent', () => {
  let component: ChatAskAQuestionComponent;
  let fixture: ComponentFixture<ChatAskAQuestionComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ ChatAskAQuestionComponent ],
      imports: [ HttpClientTestingModule, FormsModule ],
      providers: [ ChatService ]
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
