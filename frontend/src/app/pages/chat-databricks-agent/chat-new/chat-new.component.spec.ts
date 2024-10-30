import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ChatNewComponent } from './chat-new.component';
import { ChatService } from '../services/chat.service';
import { HttpClientTestingModule } from '@angular/common/http/testing';
import { ToastModule } from 'primeng/toast';

describe('ChatNewComponent', () => {
  let component: ChatNewComponent;
  let fixture: ComponentFixture<ChatNewComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ ChatNewComponent ],
      providers: [ ChatService ],
      imports: [ HttpClientTestingModule, ToastModule ]
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
