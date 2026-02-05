import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import LoginForm from '../components/forms/LoginForm';

describe('LoginForm', () => {
  beforeEach(() => {
    // @ts-ignore
    global.fetch = jest.fn();
  });

  it('submits login and calls /api/auth/login', async () => {
    // @ts-ignore
    global.fetch.mockResolvedValue({ ok: true, json: async () => ({ id: 1, email: 'a@example.com' }) });

    render(<LoginForm />);

    fireEvent.change(screen.getByLabelText(/Email/i), { target: { value: 'a@example.com' } });
    fireEvent.change(screen.getByLabelText(/Password/i), { target: { value: 'password123' } });

    fireEvent.click(screen.getByText(/Login/i));

    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledWith('/api/auth/login', expect.any(Object));
    });
  });

  it('shows error on invalid credentials', async () => {
    // @ts-ignore
    global.fetch.mockResolvedValue({ ok: false, json: async () => ({ error: 'Invalid credentials' }) });

    render(<LoginForm />);

    fireEvent.change(screen.getByLabelText(/Email/i), { target: { value: 'a@example.com' } });
    fireEvent.change(screen.getByLabelText(/Password/i), { target: { value: 'badpass' } });

    fireEvent.click(screen.getByText(/Login/i));

    await waitFor(() => {
      expect(screen.getByText(/Invalid credentials/i)).toBeInTheDocument();
    });
  });
});
