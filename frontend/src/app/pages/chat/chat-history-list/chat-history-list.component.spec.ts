import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ChatHistoryListComponent } from './chat-history-list.component';
import { ChatService } from '../services/chat.service';
import { HttpClientTestingModule } from '@angular/common/http/testing';

describe('ChatHistoryListComponent', () => {
  let component: ChatHistoryListComponent;
  let fixture: ComponentFixture<ChatHistoryListComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ ChatHistoryListComponent ],
      providers: [ ChatService ],
      imports: [ HttpClientTestingModule ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(ChatHistoryListComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
