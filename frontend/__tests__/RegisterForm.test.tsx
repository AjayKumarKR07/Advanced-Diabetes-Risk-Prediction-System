import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import RegisterForm from '../components/forms/RegisterForm';

describe('RegisterForm', () => {
  beforeEach(() => {
    // @ts-ignore
    global.fetch = jest.fn();
  });

  it('submits form and calls /api/auth/register', async () => {
    // @ts-ignore
    global.fetch.mockResolvedValue({ ok: true, json: async () => ({ id: 1, email: 'a@example.com' }) });

    render(<RegisterForm />);

    fireEvent.change(screen.getByLabelText(/Email/i), { target: { value: 'a@example.com' } });
    fireEvent.change(screen.getByLabelText(/Password/i), { target: { value: 'password123' } });
    fireEvent.change(screen.getByLabelText(/Name/i), { target: { value: 'Alice' } });

    fireEvent.click(screen.getByText(/Register/i));

    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledWith('/api/auth/register', expect.any(Object));
    });
  });

  it('shows error on failed register', async () => {
    // @ts-ignore
    global.fetch.mockResolvedValue({ ok: false, json: async () => ({ error: 'Email in use' }) });

    render(<RegisterForm />);

    fireEvent.change(screen.getByLabelText(/Email/i), { target: { value: 'a@example.com' } });
    fireEvent.change(screen.getByLabelText(/Password/i), { target: { value: 'password123' } });

    fireEvent.click(screen.getByText(/Register/i));

    await waitFor(() => {
      expect(screen.getByText(/Email in use/i)).toBeInTheDocument();
    });
  });
});
